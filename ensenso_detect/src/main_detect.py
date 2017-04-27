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

# numpy and scipy
import numpy as np
from scipy import misc

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# neural net utils
from utils import ResNet, ResidualBlock

parser = argparse.ArgumentParser(description="Process the environmental vars")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--fictitious_pub", type=bool, default=False)
parser.add_argument("--show", default=False, action="store_true")
parser.add_argument("--width", type=int, default="480")
parser.add_argument("--height", type=int, default="640")
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
            print("shows inheritance works")
        # except CvBridgeError, e:
        except CvBridgeError as e:
            rospy.logwarn ('Exception converting background ROS msg to opencv:  %s', e)
            raise
            self.img = np.zeros((320,240))

    def get_ensenso_image(self):
        return self.img

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
        self.model = ResNet(ResidualBlock, [3, 3, 3])

        def process_image(self):

            if self.args.fictitious_pub:
                self.ensenso_image = self.fictitious_pub()
            else:
                self.ensenso_image = self.get_ensenso_image()
            #Display
            # if self.args.show:
            #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #     cv2.imshow('image', self.ensenso_image)
            #     cv2.waitKey(2)

        def retrieve_net(self):

            if args.cuda():
                self.model.cuda()
            self.model.load_state_dict(torch.load('weights/' + args.net_model))





def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)
    # RS = ROS_Subscriber(args)
    proc_img = ProcessImage(args)

    #object to hold ensenso imgs
    cv_img = np.zeros(args.height, args.width)

    try:
        while not rospy.is_shuttingDown():
            cv_img = proc_img.process_image()
            rospy.sleep(10)
        # rospy.spin()

    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
