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
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--fictitious_pub", type=bool, default=False)
parser.add_argument("--net_model", type=str, default="resnet_acc=97_iter=1000.pkl")
args = parser.parse_args()

class ROS_Subscriber(object):
    def __init__(self, args):
        '''Initialize ros subscriber'''
        self.bridge = CvBridge()
        self.img = np.zeros((320,240))  #default initialization

        if args.fictitious_pub:
            self.image_pub = rospy.Publisher("/ensenso/image_combo",
               Image, queue_size=1)
            self.fictitious_pub()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/ensenso/image_combo",
            Image, self.callback,  queue_size = 1)

        if args.verbose :
            print("subscribed to /ensenso/image_combo")

    def fictitious_pub(self):
        image_np = np.array((np.random.randint(255, size=(320, 240))), dtype=np.uint8)
        image_cv = cv2.imdecode(image_np, 1)
        # image_cv = misc.toimage(image_cv)
        # image_cv = cv2.from_array(image_cv)

        cv2.imshow('cv_img', image_np)
        cv.waitKey(2)

        fictitious_msg = Image()
        fictitious_msg.header.stamp = rospy.Time.now()
        # cv2.imshow("image", image_cv)
        fictitious_msg.data = np.array(cv2.imencode('.jpg', image_cv)).tostring()

        self.image_pub.publish(fictitious_msg)
        # return image_cv

    def callback(self, rosimg):
        '''
        Callback function of subscribed topic.
        Here images get converted and features detected
        '''
        if args.verbose :
            print('received image')

        # Convert the image.
        try:
            '''
            If desired_encoding is ``"passthrough"``, then the returned image has the same format as img_msg.
            Otherwise desired_encoding must be one of the standard image encodings
            '''
            self.img = self.bridge.imgmsg_to_cv2(rosimg, 'passthrough')
        # except CvBridgeError, e:
        except CvBridgeError as e:
            rospy.logwarn ('Exception converting background ROS msg to opencv:  %s', e)
            raise
            self.img = np.zeros((320,240))

        #Display
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img)
        cv2.waitKey(2)

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)
    RS = ROS_Subscriber(args)
    try:
        if args.fictitious_pub:
            RS.fictitious_pub()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
