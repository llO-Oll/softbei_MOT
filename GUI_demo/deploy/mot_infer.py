import os
import time
import yaml
import cv2
import numpy as np
import paddle
from benchmark_utils import PaddleInferBenchmark
from preprocess import preprocess, NormalizeImage, Permute
from mot_preprocess import LetterBoxResize

from tracker import JDETracker
from ppdet.modeling.mot import visualization as mot_vis
from ppdet.modeling.mot.utils import Timer as MOTTimer

from paddle.inference import Config
from paddle.inference import create_predictor
from utils import argsparser, Timer, get_current_memory_mb
from infer import get_test_images, print_arguments

# Global dictionary
MOT_SUPPORT_MODELS = {
    'JDE',
    'FairMOT',
}


class MOT_Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN 
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.tracker = JDETracker()

    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = create_inputs(im, im_info)
        return inputs

    def postprocess(self, pred_dets, pred_embs):
        online_targets = self.tracker.update(pred_dets, pred_embs)
        online_tlwhs, online_ids = [], []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.tracker.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(tscore)
        return online_tlwhs, online_scores, online_ids

    def predict(self, image, threshold=0.5, repeats=1):
        '''
        Args:
            image (dict): dict(['image', 'im_shape', 'scale_factor'])
            threshold (float): threshold of predicted box' score
        Returns:
            online_tlwhs, online_ids (np.ndarray)
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image)
        self.det_times.preprocess_time_s.end()
        pred_dets, pred_embs = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            pred_dets = boxes_tensor.copy_to_cpu()
            embs_tensor = self.predictor.get_output_handle(output_names[1])
            pred_embs = embs_tensor.copy_to_cpu()

        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        online_tlwhs, online_scores, online_ids = self.postprocess(pred_dets,
                                                                   pred_embs)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1
        return online_tlwhs, online_scores, online_ids


def create_inputs(im, im_info):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.array((im, )).astype('float32')
    inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')
    inputs['scale_factor'] = np.array(
        (im_info['scale_factor'], )).astype('float32')
    return inputs


class PredictConfig_MOT():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in MOT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], MOT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1088,
                   trt_opt_shape=608,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16/trt_int8)
        batch_size (int): size of pre batch in inference
        use_gpu (bool): whether use gpu
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN 
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {'image': [1, 3, trt_min_shape, trt_min_shape]}
            max_input_shape = {'image': [1, 3, trt_max_shape, trt_max_shape]}
            opt_input_shape = {'image': [1, 3, trt_opt_shape, trt_opt_shape]}
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def write_mot_results(filename, results, data_type='mot'):
    if data_type in ['mot', 'mcmot', 'lab']:
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, tscores, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    score=score)
                f.write(line)


class FLAGS(object):
	def __init__(self):
		# self.model_dir = None
		self.model_dir = ".\\models"
		self.image_file = None
		self.image_dir = None
		self.batch_size = 1
		# self.video_file = None
		self.video_file = ".\\videos\\ttt.mp4"
		self.camera_id = -1
		self.threshold = 0.5
		self.output_dir = "output"
		self.run_mode = "fluid"
		self.device = "cpu"
		self.use_gpu = False
		self.run_benchmark = False
		self.enable_mkldnn = False
		self.cpu_threads = 1
		self.trt_min_shape = 1
		self.trt_max_shape = 1280
		self.trt_opt_shape = 640
		self.trt_calib_mode = False
		self.save_images = True
		self.save_results = True


class MOT_Predict(object):
    def __init__(self, FLAGS):
        paddle.enable_static()
        self.flag = FLAGS
        print_arguments(self.flag)

        self.flag = FLAGS
        pred_config = PredictConfig_MOT(self.flag.model_dir)
        self.detector = MOT_Detector(
            pred_config,
            self.flag.model_dir,
            use_gpu=self.flag.use_gpu,
            run_mode=self.flag.run_mode,
            trt_min_shape=self.flag.trt_min_shape,
            trt_max_shape=self.flag.trt_max_shape,
            trt_opt_shape=self.flag.trt_opt_shape,
            trt_calib_mode=self.flag.trt_calib_mode,
            cpu_threads=self.flag.cpu_threads,
            enable_mkldnn=self.flag.enable_mkldnn)
        self.fps = 30
        self.frame_id = 0

    def predict_via_frame(self, frame):
        online_tlwhs, online_scores, online_ids = self.detector.predict(
        frame, self.flag.threshold)

        online_im = mot_vis.plot_tracking(
            frame,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=self.frame_id,
            fps=self.fps)
        count = len(online_ids)

        return online_im,count

    def test(self):
        capture = cv2.VideoCapture(self.flag.video_file)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frame_count', frame_count)
        
        cv2.namedWindow("Tracking Detection",0);
        cv2.resizeWindow("Tracking Detection", 640, 480);
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            online_im = self.predict_via_frame(frame)
            im = np.array(online_im)

            self.frame_id += 1
            print('detect frame:%d' % (self.frame_id))

            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    my_mot_predict = MOT_Predict(FLAGS)
    my_mot_predict.test()


if __name__ == '__main__':
    # paddle.enable_static()
    # # parser = argsparser()
    # # FLAGS = parser.parse_args()
    FLAGS = FLAGS()
    # print_arguments(FLAGS)
    main()



def predict_video(detector, camera_id, FLAGS):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'mot_output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    fps = 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count', frame_count)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # yapf: disable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # yapf: enable
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = []
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        online_tlwhs, online_scores, online_ids = detector.predict(
            frame, FLAGS.threshold)
        timer.toc()

        results.append((frame_id + 1, online_tlwhs, online_scores, online_ids))
        fps = 1. / timer.average_time
        online_im = mot_vis.plot_tracking(
            frame,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=fps)
        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)),
                online_im)
        frame_id += 1
        print('detect frame:%d' % (frame_id))
        im = np.array(online_im)
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if FLAGS.save_results:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)
    writer.release()


