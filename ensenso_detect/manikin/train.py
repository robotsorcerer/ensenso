#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Olalekan Ogunmolu'

#py utils
import os
import json, time
import argparse
from PIL import Image
from os import listdir

#GPU utils
# try: import setGPU
# except ImportError: pass

#torch utils
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.nn.functional import softmax

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
parser.add_argument('--cuda', action='store_true', default=False, help="use cuda or not?")
parser.add_argument('--disp', type=bool, default=False, help="populate training samples in visdom")
parser.add_argument('--cmaxIter', type=int, default=50, help="classfier max iterations")
parser.add_argument('--num_iter', type=int, default=5)
parser.add_argument('--cbatchSize', type=int, default=1, help="classifier batch size")
parser.add_argument('--clr', type=float, default=1e-3, help="classifier learning rate")
parser.add_argument('--rnnLR', type=float, default=1e-3, help="regressor learning rate")
parser.add_argument('--classifier', type=str, default='')
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

        imagesAll = self.true_images

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
        bbox   = torch.from_numpy(self.face_left_right).double()
        bbox = bbox.unsqueeze(0).expand(1, bbox.size(0), bbox.size(1))

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
        bbox_dataset = data.TensorDataset(bbox, bbox)
        bbox_loader = data.DataLoader(bbox_dataset, batch_size=self.args.cbatchSize, shuffle=True)

        #check size of slices
        if self.args.verbose:
            print('train_X and train_Y sizes: {} | {}'.format(train_X.size(), train_Y.size()))
            print('test_X and test_Y sizes: {} | {}'.format(test_X.size(), test_Y.size()))

        return train_loader, test_loader, bbox_loader

def trainClassifierRegressor(train_loader, bbox_loader, args):
    #cnn hyperparameters
    clr = args.clr
    batchSize = args.cbatchSize
    maxIter = args.cmaxIter

    #rnn hyperparameters
    numLayers, seqLength = 2, 5
    noutputs, rlr = 12, args.rnnLR
    inputSize, nHidden = 128, [64, 32]

    resnet = ResNet(ResidualBlock, [3, 3, 3])

    #extract feture cube of last layer and reshape it
    res_classifier, feature_cube = None, None
    if args.classifier:    #use pre-trained classifier
      res_classifier = ResNet(ResidualBlock, [3, 3, 3])
      res_classifier.load_state_dict(torch.load('models225/' + args.classifier))
      print('using pretrained model')
    #   #freeze optimized layers
      for param in res_classifier.parameters():
          param.requires_grad = False
      res_classifier

    #determine classification loss and clsfx_optimizer
    clsfx_crit = nn.CrossEntropyLoss()
    clsfx_optimizer = torch.optim.Adam(resnet.parameters(), clr)

    last_layer, feat_cube = resnet.fc, []
    #accummulate all the features of the fc layer into a list
    for param in last_layer.parameters():
        feat_cube.append(param)  #will contain weights and biases
    regress_input, params_bias = feat_cube[0], feat_cube[1]

    #reshape regress_input
    regress_input = regress_input.view(-1)

    X_tr = int(0.8*len(regress_input))
    X_te = int(0.2*len(regress_input))
    X = len(regress_input)

    #reshape inputs
    rtrain_X = torch.unsqueeze(regress_input, 0).expand(seqLength, 1, X)
    rtest_X = torch.unsqueeze(regress_input[X_tr:], 0).expand(seqLength, 1, X_te+1)
    # Get regressor model and predict bounding boxes
    regressor = StackRegressive(inputSize=128, nHidden=[64,32,12], noutputs=12,\
                          batchSize=args.cbatchSize, cuda=args.cuda, numLayers=2)

    targ_X = None
    for _, targ_X in bbox_loader:
        targ_X = targ_X

    if(args.cuda):
        rtrain_X = rtrain_X.cuda()
        rtest_X  = rtest_X.cuda()
        targ_X = targ_X.cuda()
        # regressor = regressor.cuda()

    #define optimizer
    rnn_optimizer = optim.SGD(regressor.parameters(), rlr)

    # Train classifier
    for epoch in range(maxIter): #run through the images maxIter times
        for i, (train_X, train_Y) in enumerate(train_loader):

            if(args.cuda):
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
                resnet  = resnet.cuda()

            images = Variable(train_X)
            labels = Variable(train_Y)
            #rnn input
            rtargets = Variable(targ_X[:,i:i+seqLength,:])
            #reshape targets for inputs
            rtargets = rtargets.view(seqLength, -1)

            # Forward + Backward + Optimize
            clsfx_optimizer.zero_grad()
            rnn_optimizer.zero_grad()

            #predict classifier outs and regressor outputs
            outputs = resnet(images)
            routputs = regressor(rtrain_X)

            #compute loss
            loss = clsfx_crit(outputs, labels)
            rloss    = regressor.criterion(routputs, rtargets)

            #backward pass
            loss.backward()
            rloss.backward()

            # step optimizer
            clsfx_optimizer.step()
            rnn_optimizer.step()

            print ("Epoch [%d/%d], Iter [%d] cLoss: %.8f, rLoss: %.4f" %(epoch+1, maxIter, i+1,
                                                loss.data[0], rloss.data[0]))

            if epoch % 5 == 0 and epoch >0:
                clr *= 1./epoch
                rlr *= 1./epoch

                clsfx_optimizer = optim.Adam(resnet.parameters(), clr)
                rnn_optimizer   = optim.SGD(regressor.parameters(), rlr)

            # if i+seqLength >= int(targ_X.size(1)):
            #     break
    return resnet, regressor, rtest_X

def testClassifierRegressor(test_loader, resnet, regressnet, rtest_X, args):
    correct, total = 0, 0

    rtest_X = rtest_X.cuda() if args.cuda else x
    for test_X, test_Y in test_loader:

        test_X = test_X.cuda() if args.cuda else test_X
        images = Variable(test_X)
        labels = test_Y

        #forward
        outputs = resnet(images)
        routputs = regressnet(rtest_X)

        #check predcictions
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    score = 100 * correct / total
    print('Accuracy of the model on the test images: %d %%' %(score))

    # Save the Models
    torch.save(resnet.state_dict(), 'resnet_'+ str(score) + str(args.cmaxIter))
    torch.save(regressnet.state_dict(), 'regressnet_' + str(args.cmaxIter))

    #test regressor
    # for i in range(rtest_X.size(2)):
    #     routputs = regressnet(rtest_X)
    #     print('ground truth bounding box: ', bbox)
    #     rpredictions =

def main(args):
    #obtain training and testing data
    lnp = LoadAndParse(args)
    train_loader, test_loader, bbox_loader = lnp.partitionData()

    # train  conv+rnn nets
    net, reg, rtest_X = \
                trainClassifierRegressor(train_loader, bbox_loader, args)

    # test conv+rnn nets
    testClassifierRegressor(test_loader, net, reg, rtest_X, args)

if __name__=='__main__':
    main(args)
