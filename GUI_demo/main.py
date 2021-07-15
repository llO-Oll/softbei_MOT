from MainForm import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QVBoxLayout
from PyQt5 import QtGui, QtCore ,QtWidgets
from PyQt5.QtGui import QPixmap

import sys
import numpy as np
import cv2
import time
import os

import paddle
from deploy.mot_infer import MOT_Predict
from deploy.infer import print_arguments


class FLAGS(object):
	def __init__(self):
		# self.model_dir = None
		self.model_dir = ".\\models"
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


class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)
	frame_percent = pyqtSignal(int)
	count = pyqtSignal(int)
	frame_num = pyqtSignal(int)

	def __init__(self,fileName,threshold,save_result, check_GPU):
		super().__init__()
		self.fileName = fileName
		self.FLAGS = FLAGS()
		self.FLAGS.video_file = fileName
		self.FLAGS.threshold = threshold
		print(f"threshold: {threshold}")
		self.mot_predict = MOT_Predict(self.FLAGS)
		self.save_result = save_result
		if check_GPU:
			self.FLAGS.use_gpu = True
		else:
			self.FLAGS.use_gpu = False

	def run(self):
		cap = cv2.VideoCapture(self.fileName[0])
		frame_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		frame_count = 1
		print(f"Total frame: {frame_total}")

		if self.save_result:
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps = 30
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			video_name = os.path.split(self.fileName[0])[-1]
			video_name = "out_" + video_name
			out_path = os.path.join(self.FLAGS.output_dir, video_name)
			writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
			print(out_path)
			while True:
				ret, cv_img = cap.read()
				if not ret:
					break
				cv_img,count_val = self.mot_predict.predict_via_frame(cv_img)
				self.change_pixmap_signal.emit(cv_img)
				print(f"frame percent:{100*frame_count/frame_total}%")
				self.frame_percent.emit(int(100*frame_count/frame_total))
				frame_count = frame_count + 1
				self.count.emit(count_val)
				self.frame_num.emit(frame_count)

				im = np.array(cv_img)
				writer.write(im)
			writer.release()
		else:
			while True:
				ret, cv_img = cap.read()
				if not ret:
					break
				cv_img,count_val = self.mot_predict.predict_via_frame(cv_img)
				self.change_pixmap_signal.emit(cv_img)
				print(f"frame percent:{100*frame_count/frame_total}%")
				self.frame_percent.emit(int(100*frame_count/frame_total))
				frame_count = frame_count + 1
				self.count.emit(count_val)
				self.frame_num.emit(frame_count)


class MainForm(QMainWindow):
	def __init__(self):
		super(MainForm,self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.disply_width = 900
		self.display_height = 700
		# create the label that holds the image
		# self.video_frame = QtWidgets.QLabel(self)
		# self.video_frame.resize(self.disply_width, self.display_height)

		self.ui.exit_button.clicked.connect(QtCore.QCoreApplication.quit)
		self.ui.start_button.clicked.connect(self.start_button_pressed)
		self.ui.open_button.clicked.connect(self.open_button_pressed)
		self.ui.recog_prog.setValue(0)

		self.ui.conf_val.setValue(0.5)
		# print(f"conf val: {self.ui.conf_val.value()}")

	def open_button_pressed(self):
		self.fileName = QFileDialog.getOpenFileName(None,caption="",directory=QtCore.QDir.currentPath())
		print(self.fileName[0])

	def start_button_pressed(self):
		# create the video capture thread
		self.thread = VideoThread(self.fileName,self.ui.conf_val.value(),self.ui.save_result.isChecked(),self.ui.check_gpu.isChecked())
		# connect its signal to the update_image slot
		self.thread.change_pixmap_signal.connect(self.update_image)
		# connect its signal to change the progress bar
		self.thread.frame_percent.connect(self.recog_prog_change)
		self.thread.count.connect(self.count_change)
		self.thread.frame_num.connect(self.frame_count_change)
		# start the thread
		self.thread.start()

	def frame_count_change(self, val):
		self.ui.frame_count_val.setText(str(val))

	def recog_prog_change(self, val):
		# print(f"val:{val}")
		self.ui.recog_prog.setValue(val)

	def count_change(self, val):
		self.ui.count_val.setText(str(val))

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.ui.video_frame.setPixmap(qt_img)

	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)


app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = MainForm()
ui.show()
sys.exit(app.exec_())