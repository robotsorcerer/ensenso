#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

"""OpenCV loader of pcl topic
"""
__author__ = 'Olalekan Ogunmolu <Olalekan.Ogunmolu@utdallas.edu>'
__version__ = '0.1'
__license__ = 'MIT'


import sys, time
import argparse
import json, os

# numpy and scipy
import numpy as np
from scipy import misc

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
import rospkg

# Ros Messages
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# neural net utils
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from utils import ResNet, ResidualBlock

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
	 color_scheme='Linux', call_pdb=1)

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	if v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Process the environmental vars")
parser.register('type','bool',str2bool) # add type keyword to registries
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--cuda", default=True, action="store_true")
parser.add_argument("--fictitious_pub", type=bool, default=False)
parser.add_argument("--show", default=True, action="store_true")
parser.add_argument("--width", type=int, default="1024", help="default spatial dim from ids")
parser.add_argument("--height", type=int, default="1280", help="default spatial dim from ids")
parser.add_argument("--net_model", type=str, default="resnet_acc=97_iter=1000.pkl")
args = parser.parse_args()

class ROS_Subscriber(object):
	def __init__(self, args):
		'''Initialize ros subscriber'''
		self.args = args
		self.bridge = CvBridge()
		self.img = np.zeros((args.height, args.width))  #default initialization

		if self.args.fictitious_pub:
			self.image_pub = rospy.Publisher("/ensenso/image_combo",
			   Image, queue_size=1)

		# subscribed Topic
		self.subscriber = rospy.Subscriber("/ensenso/image_combo",
			Image, self.callback,  queue_size = 1)

		if self.args.verbose :
			print("subscribed to /ensenso/image_combo")

	def fictitious_pub(self):
		image_np = np.array((np.random.randint(255,
					size=(self.args.height, self.args.width))), dtype=np.uint8)
		image_cv = cv2.imdecode(image_np, 1)

		if self.args.show:
			cv2.imshow('cv_img', image_np)
			cv.waitKey(2)

		fictitious_msg = Image()
		fictitious_msg.header.stamp = rospy.Time.now()
		fictitious_msg.data = np.array(cv2.imencode('.jpg', image_cv)).tostring()

		self.image_pub.publish(fictitious_msg)

	def callback(self, rosimg):
		'''
		Callback function of subscribed topic.
		Here images get converted and features detected
		'''
		if self.args.verbose :
			print('received image')

		# Convert the image.
		try:
			'''
			If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
			Otherwise desired_encoding must be one of the standard image encodings
			'''
			self.img = self.bridge.imgmsg_to_cv2(rosimg, 'passthrough')
			# print("shows inheritance works")
		# except CvBridgeError, e:
		except CvBridgeError as e:
			rospy.logwarn ('Exception converting background ROS msg to opencv:  %s', e)
			raise
			self.img = np.zeros((320,240))

	def get_ensenso_image(self):
		# print('called get_ensenso_image')
		return self.img

	def process_image(self):
		"""
		Override this method in derived class
		"""

	def classify(self):
		"override this in process images"

class ProcessImage(ROS_Subscriber):
	"""
	Retrieve opencv image via ros topic and
	classify if image contains face or not as
	defined by our pretrained convnet model.

	Inherits from class ROS_Subscriber
	"""

	def __init__(self, args):
		ROS_Subscriber.__init__(self, args)
		self.args = args
		self.counter = 0

		self.preprocess = transforms.Compose([
		   transforms.ToPILImage(),
		   transforms.Scale(40),
		   transforms.RandomHorizontalFlip(),
		   transforms.RandomCrop(32),
		   transforms.ToTensor()
		])

		labels_path = self.getPackagePath('ensenso_detect') + '/manikin/labels.json'
		self.classes = json.loads(open(labels_path).read()) # 0 = fake, 1=real

	def getPackagePath(self, package_name):
		rospack = rospkg.RosPack()
		path = rospack.get_path(package_name)
		return path


	def process_image(self):

		if self.args.fictitious_pub:
			self.ensenso_image = self.fictitious_pub()
		else:
			self.ensenso_image = self.get_ensenso_image()

		'''convert retrieved open cv image to torch tensors'''
		# first allocate tensor storage object
		self.rawImgTensor = torch.LongTensor(1024, 1280)
		# then copy imgs over
		self.rawImgTensor = torch.from_numpy(self.ensenso_image)
		self.raw_size = [int(x) for x in self.rawImgTensor.size()]
		# we need to resize the raw image to the size of the trained net
		self.rawImgTensor = self.rawImgTensor.unsqueeze(0).expand(3, self.raw_size[0], self.raw_size[1])
		#convert the tensor to PIL image and back for easy proc by transforms
		self.rawImgTensor = self.preprocess(self.rawImgTensor.float())
		self.rawImgTensor = self.rawImgTensor.unsqueeze(0)

	def classify(self):

		#pre-pro images first
		self.process_image()
			# Test
		correct, total = 0, 0
		resnet = self.args.net
		test_Y = torch.from_numpy(np.array([0, 1]))#, 0)) #self.loadLabelsFromJson()
		test_X = self.rawImgTensor.double()

		if(args.cuda):
			test_X = test_X.cuda()
		images = Variable(test_X)

		labels = test_Y
		outputs = resnet(images)
		_, predicted = torch.max(outputs, 1)

		#collect classes
		classified = predicted.data[0][0]
		index = int(classified)

		img_class = self.classes[str(index)]

		#Display
		if self.args.show:
			cv2.namedWindow('image', cv2.WINDOW_NORMAL)
			#putText Properties
			org = img_class + ' face'
			cv2.putText(self.ensenso_image, org, (15, 55), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 3), thickness=2, lineType=cv2.LINE_AA)
			cv2.imshow('image', self.ensenso_image)

			ch = 0xFF & cv2.waitKey(5)
			if ch == 27:
				rospy.signal_shutdown("rospy is shutting down")

		# print('I see a {} face'.format(self.classes[str(index)]))


	def retrieve_net(self, model):
		# get ensenso_detect path
		detect_package_path = self.getPackagePath('ensenso_detect')
		weightspath = detect_package_path + '/manikin/models225/'

		base, ext = os.path.splitext(args.net_model)
		if (ext == ".pkl"):  #using high score model
			if args.cuda:
				model.cuda()
			model.load_state_dict(torch.load(weightspath + args.net_model))
		elif(ext ==".pth"):
			model = torch.load(weightspath + args.net_model)
			if args.cuda:
				model.cuda()
			model.eval()
		else:
			rospy.logwarn("supplied neural net extension is unknown")

		return model

def main(args):
	'''Initializes and cleanup ros node'''
	rospy.init_node('image_feature', anonymous=True)

	proc_img = ProcessImage(args)
	model = ResNet(ResidualBlock, [3, 3, 3])
	model = proc_img.retrieve_net(model)

	if model:
		args.net = model

	try:
		rate = rospy.Rate(30)
		while not rospy.is_shutdown():
			proc_img.classify()
			rate.sleep()

	except KeyboardInterrupt:
		print ("Shutting down ROS Image feature detector module")

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(args)
