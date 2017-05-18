#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

"""OpenCV loader of pcl topic
   ConvNet Classifier of Manikin Face
   Detector of Face Features
"""
__author__ = 'Olalekan Ogunmolu <Olalekan.Ogunmolu@utdallas.edu>'
__version__ = '0.1'
__license__ = 'MIT'


import sys, time
import argparse
import json, os
import visdom
import sys
import os.path as osp

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
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

# from utils import ResNet, ResidualBlock
this_dir = osp.dirname(__file__)
models_path = osp.join(this_dir, '..', 'manikin')#, 'model')
sys.path.insert(0, models_path) if models_path not in sys.path else None

from model import ResNet, ResidualBlock, StackRegressive

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
parser.add_argument("--reg_net", type=str, default="regressnet_iter50.pkl")
parser.add_argument("--conv_net", type=str, default="resnet_score100.pkl")
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
        self.vis = visdom.Visdom()
        self.weights = None
        self.seqLength = 5
        #weights and bias for classifier's fully-connected layer
        self.fc_bias, self.fc_weights = None, None

        self.normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )

        self.preprocess = transforms.Compose([
           transforms.ToPILImage(),
           transforms.Scale(40),
           transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32),
           transforms.ToTensor()
        #    self.normalize
        ])

        labels_path = self.getPackagePath('ensenso_detect') + '/manikin/labels.json'
        self.classes = json.loads(open(labels_path).read()) # 0 = fake, 1=real

    def getPackagePath(self, package_name):
        rospack = rospkg.RosPack()
        path = rospack.get_path(package_name)
        return path

    def retrieve_net(self, conv_model, regress_model):
        # get ensenso_detect path
        detect_package_path = self.getPackagePath('ensenso_detect')
        conv_path = detect_package_path + '/manikin/models225/'
        weightspath = detect_package_path + '/manikin/models_new/'

        base, ext = os.path.splitext(args.conv_net)

        if (ext == ".pkl"):         #using high score model
            conv_model.load_state_dict(torch.load(conv_path + args.conv_net))
            regress_model.load_state_dict(torch.load(weightspath + args.reg_net))

            # set up models for evaluation mode
            conv_model.eval()
            regress_model.eval()
        else:
            conv_model = torch.load(weightspath + args.conv_net)
            regress_model = torch.load(weightspath + args.reg_net)

            reg_net.eval()
            conv_model.eval()
        return conv_model, regress_model

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
        raw_size = [int(x) for x in self.rawImgTensor.size()]
        # we need to resize the raw image to the size of the trained net
        self.rawImgTensor = self.rawImgTensor.unsqueeze(0).expand(3, raw_size[0], raw_size[1])
        #convert the tensor to PIL image and back for easy proc by transforms
        self.rawImgTensor = self.preprocess(self.rawImgTensor.float())
        self.rawImgTensor = self.rawImgTensor.unsqueeze(0)

    def classify_and_draw(self, conv_model, regress_model):
        #pre-pro images first
        total, correct = 0,0
        self.process_image()
        test_X = self.rawImgTensor.double()

        if(args.cuda):
            test_X = test_X.cuda()
        images = Variable(test_X)

        conv_model.eval()
        regress_model.eval()

        outputs = conv_model(images)
        _, predicted = torch.max(outputs.data, 1)

        #extract input for regressor
        last_layer, feat_cube = conv_model.fc, []
        #accummulate all the features of the fc layer into a list
        for param in last_layer.parameters():
            feat_cube.append(param)  #will contain weights and biases
        regress_input, params_bias = feat_cube[0], feat_cube[1]

        #reshape regress_input
        regress_input = regress_input.view(-1)
        rtrain_X = torch.unsqueeze(regress_input, 0).expand(self.seqLength, 1, len(regress_input))

        rtrain_X = rtrain_X.cuda() if self.args.cuda else rtrain_X

        # now forward last fc layer through regressor net to get bbox predictions
        bounding_box = regress_model(rtrain_X).data[0]

        '''
        The predicted bounding boxes are are follows:

        First 4 cols represent top and lower coordinates of face boxes,
        Followed by 2 cols belonging to left eye pixel coords,
        last 2 cols are the right eye coords
        '''
        bbox = []
        for i in range(bounding_box.size(0)):
            bbox.append(int(bounding_box[i]))

        # print('bounding box: ', bbox)#, bounding_box[0])

        #this is to allow us see the weights in visdom
        self.weights = conv_model.state_dict().items() #
        self.fc_weights = conv_model.state_dict()['fc.weight']
        self.fc_bias = conv_model.state_dict()['fc.bias']
        #convert the fc weights to np array in order to see the activations in visdom
        fc_weights_np = self.fc_weights.cpu().numpy()
        if args.verbose:
            print('fc_weights: ', fc_weights_np.shape)  # should be torch.cuda.DoubleTensor of size 64
        self.vis.image(fc_weights_np)

        #collect classes
        index = int(predicted[0][0])
        img_class = self.classes[str(index)]

        #Display
        if self.args.show:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            org = img_class + ' face'

            cv2.putText(self.ensenso_image, org, (15, 55), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 150, 0), thickness=2, lineType=cv2.LINE_AA)
            #draw bounding boxes
            cv2.rectangle(self.ensenso_image, (bbox[0], bbox[1]), (bbox[6], bbox[7]), (255, 60, 10), 2)
            cv2.circle(self.ensenso_image, (bbox[8], bbox[9]), 2, (255, 0, 0), 3, 8)
            cv2.circle(self.ensenso_image, (bbox[10], bbox[11]), 2, (0, 0, 255), 3, 8)
            cv2.imshow('image', self.ensenso_image)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                rospy.signal_shutdown("rospy is shutting down")

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)

    proc_img = ProcessImage(args)
    conv_model = ResNet(ResidualBlock, [3, 3, 3]).cuda()
    regress_model = StackRegressive(inputSize=128, nHidden=[64,32,12], noutputs=12,\
                          batchSize=1, cuda=args.cuda, numLayers=2)
    conv_model, regress_model = proc_img.retrieve_net(conv_model, regress_model)

    if conv_model:
        args.net = conv_model

    try:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            proc_img.classify_and_draw(conv_model, regress_model)
            rate.sleep()

    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(args)
