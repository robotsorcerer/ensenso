#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Olalekan Ogunmolu'

#py utils
import os
import json
import argparse
from PIL import Image
from os import listdir

#GPU utils
# try: import setGPU
# except ImportError: pass

#torch utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

#cv2/numpy utils
import cv2
import numpy as np
import numpy.random as npr
from random import shuffle

import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

#myne utils
from model import ResNet, ResidualBlock, StackRegressive
from utils import get_bounding_boxes as bbox

parser = argparse.ArgumentParser(description='Process environmental variables')
parser.add_argument('--cuda', action='store_true', help="use cuda or not?")
parser.add_argument('--disp', type=bool, default=False, help="populate training samples in visdom")
parser.add_argument('--cmaxIter', type=int, default=200, help="classfier max iterations")
parser.add_argument('--num_iter', type=int, default=5)
parser.add_argument('--cbatchSize', type=int, default=1, help="classifier batch size")
parser.add_argument('--clr', type=float, default=1e-3, help="classifier learning rate")
parser.add_argument('--rnnLR', type=float, default=5e-3, help="regressor learning rate")
parser.add_argument('--classifier', type=str, default='resnet_acc=97_iter=1000.pkl')
parser.add_argument('--cepoch', type=int, default=500)
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()
print(args)
torch.set_default_tensor_type('torch.DoubleTensor')

class LoadAndParse(object):

    def __init__(self, args, true_path="raw/face_images/", fake_path="raw/face_neg/"):
        '''
        from:
        https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L108-L113
        https://github.com/pytorch/examples/blob/409a7262dcfa7906a92aeac25ee7d413baa88b67/imagenet/main.py#L94-L95
        '''
        self.args = args
        self.normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
           transforms.Scale(40),
           transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32),
           transforms.ToTensor()
        #    self.normalize
        ])

        #provide path to true and fake images
        self.true_path= true_path

        pwd = os.getcwd()
        face_neg = pwd + "/raw/" + "face_neg"
        face_neg_1 = face_neg + '/' + 'neg_1'
        face_neg_2 = face_neg + '/' + 'neg_2'
        face_neg_3 = face_neg + '/' + 'neg_3'
        face_neg_4 = face_neg + '/' + 'neg_4'

        self.neg_dirs = [face_neg_1, face_neg_2, face_neg_3, face_neg_4]
        self.fake_path= fake_path
        self.real_images = []

        self.true_images = None
        self.faces_bbox = None
        self.left_bbox = None
        self.right_bbox = None
        self.face_left_right = None
        #define tensors to hold the images in memory
        self.real_images, self.real_labels  = [], []


    # #load labels file
    def loadLabelsFromJson(self):
        labels_file = open('labels.json').read()
        labels = json.loads(labels_file)
        classes = labels  # 0 = fake, 1=real
        return classes

    def loadImages(self, path):
        # return array of images
        imagesList = listdir(path)

        loadedImages, faces_bbox = [], []
        left_bbox, right_bbox = [], []

        #bounding boxes
        dict_combo = bbox()
        faces_dict, left_dict, right_dict = dict_combo[0], dict_combo[1],  dict_combo[2]

        #load serially to ensure labels match
        for image in imagesList:
            img = Image.open(path + image)
            face = faces_dict[image]
            left = left_dict[image]
            right = right_dict[image]

            loadedImages.append(img)
            faces_bbox.append(face)
            left_bbox.append(left)
            right_bbox.append(right)

        return loadedImages, faces_bbox, left_bbox, right_bbox

    def loadNegative(self):
        negative_list = []
        for dirs in self.neg_dirs:
            for img_path in listdir(dirs):
                base, ext = os.path.splitext(img_path)
                if ext == '.jpg':
                    img = Image.open(dirs + '/' + img_path)
                    negative_list.append(img)
                    if self.args.verbose:
                        print('appending {} to {} list'.format(img_path, 'negative'))
        return negative_list

    # get images in the dir
    def getImages(self):
        #load images
        self.true_images, self.faces_bbox, self.left_bbox,self.right_bbox = self.loadImages(self.true_path)
        # concatenate to ease numpy issues
        left_right = np.concatenate((self.left_bbox, self.right_bbox), axis=1)

        faces_top, faces_bot = [], []
        #arrange faces_bbox into a 1x8 array
        for i in range(len(self.faces_bbox)):
            faces_top.append(self.faces_bbox[i][0])
            faces_bot.append(self.faces_bbox[i][1])

        """
        First 4 cols represent top coordinates of face boxes,
        Followed by lower coordinates of face_boxes
        Next 2 cols belong to left eye centers, last col are right eye coords
        """
        self.face_left_right = np.concatenate((faces_top, faces_bot, left_right), axis=1)

        #define labels
        self.real_labels = [1]*len(self.true_images)  #faces
        # self.fake_labels = [0]*len(fake_images)  #backgrounds

        imagesAll = self.true_images #+ fake_images

        # Now preprocess and create list for images
        for imgs in imagesAll:
            images_temp = self.preprocess(imgs).double()

            #Take care of non-singleton dimensions in negative images
            if not images_temp.size(0) == 3:
                images_temp = images_temp.expand(3, images_temp.size(1), images_temp.size(2))
            self.real_images.append(images_temp)

    def partitionData(self):
        # retrieve the images first
        self.getImages()

        #Now separate true and fake to training and testing sets
        portion_train = 0.8
        portion_test = 0.2
        X_tr = int(portion_train * len(self.real_images))
        X_te = int(portion_test  * len(self.real_images))

        # allocate tensors memory
        train_X = torch.LongTensor(X_tr, self.real_images[0].size(0), self.real_images[0].size(1),
                                  self.real_images[0].size(2))
        test_X = torch.LongTensor(X_te, self.real_images[0].size(0), self.real_images[0].size(1),
                                  self.real_images[0].size(2))

        #Now copy tensors over
        train_X = torch.stack(self.real_images[:X_tr], 0)
        train_Y = torch.from_numpy(np.array(self.real_labels[:X_tr]))

        # bounding box data
        bbox_X = torch.from_numpy(self.face_left_right[:X_tr]).double()
        bbox_Y = torch.from_numpy(self.face_left_right[X_tr:]).double()

        #testing set
        test_X = torch.stack(self.real_images[X_tr:], 0)
        test_Y = torch.from_numpy(np.array(self.real_labels[X_tr:]))

        #data loaders
        train_dataset =  data.TensorDataset(train_X, train_Y)
        train_loader  =  data.DataLoader(train_dataset,
                        batch_size=self.args.cbatchSize, shuffle=True)

        #test loader
        test_dataset = data.TensorDataset(test_X, test_Y)
        test_loader = data.DataLoader(test_dataset,
                            batch_size=self.args.cbatchSize, shuffle=True)

        #bbox loader
        bbox_loader = { 'bbox_X': bbox_X, 'bbox_Y': bbox_Y }

        #check size of slices
        if self.args.verbose:
            print('train_X and train_Y sizes: {} | {}'.format(train_X.size(), train_Y.size()))
            print('test_X and test_Y sizes: {} | {}'.format(test_X.size(), test_Y.size()))

        return train_loader, test_loader, bbox_loader

def trainClassifier(train_loader, resnet, args):
    #hyperparameters
    lr = args.clr
    batchSize = args.cbatchSize
    maxIter = args.cmaxIter

    #determine classification loss and clsfx_optimizer
    clsfx_crit = nn.CrossEntropyLoss()
    clsfx_optimizer = torch.optim.Adam(resnet.parameters(), lr)

    # Train classifier
    for epoch in range(maxIter): #run through the images maxIter times
        for i, (train_X, train_Y) in enumerate(train_loader):

            if(args.cuda):
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
                resnet  = resnet.cuda()

            # images = Variable(train_X[i:i+batchSize,:,:,:])
            # labels = Variable(train_Y[i:i+batchSize,])
            images = Variable(train_X)
            labels = Variable(train_Y)

            # Forward + Backward + Optimize
            clsfx_optimizer.zero_grad()
            outputs = resnet(images)
            loss = clsfx_crit(outputs, labels)
            loss.backward()
            clsfx_optimizer.step()

            # if(epoch %2 == 0):
            # print(enumerate(train_loader)())
            print ("Epoch [%d/%d], Iter [%d] Loss: %.8f" %(epoch+1, maxIter, i+1, loss.data[0]))

            # Decaying Learning Rate
            # if (epoch+1) % 50 == 0:
            #     lr /= 3
            #     clsfx_optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    return resnet

def testClassifier(test_loader, resnet, args):
    correct, total = 0, 0
    for test_X, test_Y in test_loader:
        if(args.cuda):
            test_X = test_X.cuda()
        images = Variable(test_X)
        labels = test_Y
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    score = 100 * correct / total

    print('Accuracy of the model on the test images: %d %%' %(score))

    # Save the Model
    torch.save(resnet.state_dict(), 'resnet_'+ str(score) + str(args.cmaxIter))

def trainRegressor(args, bbox_loader):
    r"""
    Following the interpretable learning from self-driving examples:
    https://arxiv.org/pdf/1703.10631.pdf   we can extract the last
    feature cube x_t from the resnet model as a set of L = W x H
    vectors of depth D, and stack a regressor module to obtain
    bounding boxes
    """
    #hyperparameters
    inputSize = 64
    nHidden = [64, 32, 16]
    noutputs = 12
    batchSize = args.cbatchSize
    numLayers = 2
    lr=args.rnnLR
    maxIter = args.cmaxIter

    '''
    #extract feture cube of last layer and reshape it
    res_classifier = ResNet(ResidualBlock, [3, 3, 3])

    if args.classifier is not None:    #use pre-trained classifier
          res_classifier.load_state_dict(torch.load('models225/' + args.classifier))

    # Get everything but the classifier fc (last) layer
    res_cube = list(res_classifier.children()).pop()
    #reshape last layer for input of bounding box coords
    res_cube.append(nn.Linear(inputSize), inputSize)
    '''

    bbox_X = bbox_loader['bbox_X']
    bbox_Y = bbox_loader['bbox_Y']

    if(args.cuda):
        bbox_X = bbox_X.cuda()
        bbox_Y = bbox_Y.cuda()
        regressor = regressor#.cuda()

    #extract feture cube of last layer and reshape it
    res_classifier = ResNet(ResidualBlock, [3, 3, 3])

    if args.classifier is not None:    #use pre-trained classifier
          res_classifier.load_state_dict(torch.load('models225/' + args.classifier))
          res_classifier = nn.Sequential(*list(res_classifier.children()))

          #freeze optimized layers
          for param in res_classifier.parameters():
              param.requires_grad = False

    #   res_cube = list(res_classifier.children())[:-1]
    params_list = []
    #accumalate all the features of the fc layer into a list
    for p in res_classifier.fc.parameters():
        params_list.append(p)  #will contain weighs and biases

    params_weight, params_bias = params_list[0], params_list[1]

    #reshape params_weight
    params_weight = params_weight.view(128, -1)
    #xavier initialize recurrent layer

    # Get regressor model and predict bounding boxes
    regressor = StackRegressive(res_cube=res_classifier, inputSize=128, nHidden=[64,32,12], noutputs=12,\
                          batchSize=args.cbatchSize, cuda=args.cuda, numLayers=2)
    #define optimizer
    optimizer = optim.SGD(regressor.parameters(), lr)

    # Forward + Backward + Optimize
    for epoch in xrange(maxIter): #run through the images maxIter times
        # bbox_Y = Variable(bbox_Y[i, i+batchSize,])

        for i in xrange(len(bbox_X)):
            print(res_cube)
            #reshape last layer for input of bounding box coords
            # res_cube.append(nn.Linear(inputSize, inputSize))

            # images = Variable(train_X[i:i+batchSize,:,:,:])
            # labels = Variable(train_Y[i:i+batchSize,])
            # print(bbox_X[i:i+batchSize])
            outputs = regressor(res_cube)
            targets = Variable(params_weight)

            loss    = regressor.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # if (epoch % 10) == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, epoch+batchSize, bbox_X.data.size(0),
                float(i+batchSize)/bbox_X.data.size(0)*100,
                loss.data[0]))

def main(args):
    #obtain training and testing data
    lnp = LoadAndParse(args)
    train_loader, test_loader, bbox_loader = lnp.partitionData()

    #obtain classification model
    resnet = ResNet(ResidualBlock, [3, 3, 3])

    # train classifier
    # net = trainClassifier(train_loader, resnet, args)

    # test classifier and save classifier model
    # testClassifier(test_loader, resnet, args)

    #train regrerssor
    # bbox = lnp.face_left_right
    trainRegressor(args, bbox_loader)

    # Now stack regressive model on top of feature_cube of classifier

if __name__=='__main__':
    main(args)
