import paddle
from deploy.mot_infer import MOT_Predict
from deploy.infer import print_arguments
# from deploy.mot_infer import PredictConfig_MOT, MOT_Detector, predict_video


class FLAGS(object):
	def __init__(self):
		self.model_dir = None
		# self.model_dir = ".\\models"
		self.image_file = None
		self.image_dir = None
		self.batch_size = 1
		# self.video_file = None
		self.video_file = ".\\videos\\ttt.mp4"
		# self.camera_id = -1
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


def main():
	my_mot_predict = MOT_Predict(FLAGS)
	my_mot_predict.test()

if __name__ == '__main__':
    # paddle.enable_static()
    FLAGS = FLAGS()
    FLAGS.model_dir = ".\\models"
    # print_arguments(FLAGS)
    main()