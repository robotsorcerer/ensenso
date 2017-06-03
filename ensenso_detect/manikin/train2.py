#!/usr/bin/env python
from __future__ import print_function
__author__ = 'Olalekan Ogunmolu'

#py utils
import os, gc
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
import torch.nn.functional as F
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
from math import floor, exp

import sys, traceback
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

#myne utils
from model import ResNet, ResidualBlock, \
                StackRegressive, RecurrentModel
from utils import get_bounding_boxes as bbox

parser = argparse.ArgumentParser(description='Process environmental variables')
parser.add_argument('--ship2gpu', action='store_false', default=True, help="use cuda or not?")
parser.add_argument('--multinomial', action='store_true', default=True, help="sample from feature cube?")
parser.add_argument('--disp', type=bool, default=False, help="populate training samples in visdom")
parser.add_argument('--cmaxIter', type=int, default=50, help="classfier max iterations")
parser.add_argument('--num_iter', type=int, default=5)
parser.add_argument('--cbatchSize', type=int, default=1, help="classifier batch size")
parser.add_argument('--clr', type=float, default=1e-3, help="classifier learning rate")
parser.add_argument('--rnnLR', type=float, default=1e-2, help="regressor learning rate")
parser.add_argument('--classifier', type=str, default='')
parser.add_argument('--cepoch', type=int, default=500)
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

print(args)

torch.set_default_tensor_type('torch.DoubleTensor')
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True


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
        face_neg_5 = face_neg + '/' + 'neg_5'

        self.neg_dirs = [face_neg_5]
        self.fake_path= fake_path

        self.left_bbox, self.right_bbox = None, None
        self.face_left_right, self.faces_bbox = None, None

        #define tensors to hold the images in memory
        self.real_images, self.real_labels  = [], []
        self.real_labels, self.fake_labels  = [], []
        self.fakenreal_images, self.fakenreal_labels = [], []

    # #load labels file
    def loadLabelsFromJson(self):
        labels_file = open('labels.json').read()
        labels = json.loads(labels_file)
        classes = labels  # 0 = fake, 1=real
        return classes

    def loadImages(self, path):
        #load negative training images
        def loadNegative():
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

        # return array of positive images
        imagesList = listdir(path)
        posImages, faces_bbox = [], []
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

            posImages.append(img)
            faces_bbox.append(face)
            left_bbox.append(left)
            right_bbox.append(right)
        negImages = loadNegative()
        return posImages, negImages, faces_bbox, left_bbox, right_bbox

    # get images in the dir
    def getImages(self):
        #load images
        self.real_images, self.fake_images, self.faces_bbox, \
                self.left_bbox,self.right_bbox = self.loadImages(self.true_path)
        # concatenate to ease numpy issues
        left_right = np.concatenate((self.left_bbox, self.right_bbox), axis=1)

        faces_top, faces_bot = [], []
        #arrange faces_bbox into a 1x8 array
        # we only extract the top left and bot right
        for i in range(len(self.faces_bbox)):
            faces_top.append(self.faces_bbox[i][0][0:2]) #top left
            faces_bot.append(self.faces_bbox[i][1][2:4]) #bot right
        """
        First 4 cols represent top coordinates of face boxes,
        Followed by lower coordinates of face_boxes
        Next 2 cols belong to left eye centers, last col are right eye coords
        """
        self.face_left_right = np.concatenate((faces_top, faces_bot, left_right), axis=1)

        #define labels
        self.real_labels = [1]*len(self.real_images)  #faces
        self.fake_labels = [0] * len(self.fake_images) #negative faces

        imagesAll = self.real_images + self.fake_images
        labelsAll = self.real_labels + self.fake_labels

        #permute images and labels in place
        temp = list(zip(imagesAll, labelsAll))
        shuffle(temp)
        imagesAll, labelsAll = zip(*temp)

        classes = self.loadLabelsFromJson()

        # Now preprocess and create list for images
        for imgs in imagesAll:
            images_temp = self.preprocess(imgs).double()

            #Take care of non-singleton dimensions in negative images
            if not images_temp.size(0) == 3:
                images_temp = images_temp.expand(3, images_temp.size(1), images_temp.size(2))
            self.fakenreal_images.append(images_temp)
        self.fakenreal_labels = labelsAll

    def partitionData(self):
        # retrieve the images first
        self.getImages()

        #Now separate true and fake to training and testing sets
        portion_train = 0.8
        portion_test = 0.2
        X_tr = int(portion_train * len(self.fakenreal_images))
        X_te = int(portion_test  * len(self.fakenreal_images))

        # allocate tensors memory
        train_X = torch.LongTensor(X_tr, self.fakenreal_images[0].size(0), self.fakenreal_images[0].size(1),
                                  self.fakenreal_images[0].size(2))
        test_X = torch.LongTensor(X_te, self.fakenreal_images[0].size(0), self.fakenreal_images[0].size(1),
                                  self.fakenreal_images[0].size(2))

        #Now copy tensors over
        train_X = torch.stack(self.fakenreal_images[:X_tr], 0)
        train_Y = torch.from_numpy(np.array(self.fakenreal_labels[:X_tr]))

        # bounding box data
        bbox   = torch.from_numpy(self.face_left_right).double()
        bbox = bbox.unsqueeze(0).expand(1, bbox.size(0), bbox.size(1))

        #testing set
        test_X = torch.stack(self.fakenreal_images[X_tr:], 0)
        test_Y = torch.from_numpy(np.array(self.fakenreal_labels[X_tr:]))

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

resnet = ResNet(ResidualBlock, [3, 3, 3])
regressor = StackRegressive(inputSize=3276, \
                            nHidden=[int(3276/40), \
                            int(3276/120)],noutputs=8, \
                            batchSize=args.cbatchSize, ship2gpu=args.ship2gpu, \
                            numLayers=1)
rnn_optimizer = optim.Adam(regressor.parameters(), args.rnnLR)

def get_lstm_input(x, y):
    return torch.mul(x[0], y.view(1,1,-1))

def memory_usage():
        return int(open('/proc/self/statm').read().split()[1])

def select_regress_input(last_layer):
    """
    This section is a variant of this paper:
    Sharma, S., Kiros, R., & Salakhutdinov, R. (n.d.).
    ACTION RECOGNITION USING VISUAL ATTENTION. Retrieved from
    https://arxiv.org/pdf/1511.04119.pdf

    An intuitive way of reasoning about this is that we use the network and lstms to
    compute the location of possible features in the image.

    We then show the soft-max embedded features the input bounding boxes and compute the
    intersection over unions
    """

    # #extract last convolution layer
    feat_cube = []
    for param in last_layer.parameters():
        if param.dim() > 1:  # extract only conv cubes
            feat_cube.append(param)

    #max pool the last conv layer
    kernel, stride = 3, 2
    m = nn.MaxPool2d(kernel, stride)
    feat_cube[0] = m(feat_cube[0]) #will now be 64 x 32 x 1 x 1
    feat_cube[1] = m(feat_cube[1]) #will now be 64 x 64 x 1 x 1
    feat_cube[2] = m(feat_cube[2]) #will now be 64 x 32 x 1 x 1
    feat_cube[3] = m(feat_cube[3]) #will now be 64 x 64 x 1 x 1
    feat_cube[4] = m(feat_cube[4]) #will now be 64 x 64 x 1 x 1
    feat_cube[5] = m(feat_cube[5]) #will now be 64 x 64 x 1 x 1
    feat_cube[6] = m(feat_cube[6]) #will now be 64 x 64 x 1 x 1

    lt = []  # this contains the soft max attention for each pooled layer
    for x in xrange(len(feat_cube)):
        lt.append(softmax(feat_cube[x].view(1, 1, -1)))

    inLSTM1 = get_lstm_input(lt, feat_cube[0])  #will have 2048 connections
    regress = RecurrentModel(inputSize=inLSTM1.size(2), nHidden=[inLSTM1.size(2),1024, 64*32],\
                             noutputs=64*32,batchSize=args.cbatchSize, ship2gpu=args.ship2gpu, \
                             numLayers=1)
    #use normal initialization for regression layer

    if args.ship2gpu:
        regress = regress.cuda()
    gc.collect()

    for name, weights in regress.named_parameters():
        init.uniform(weights, 0, 1)

    y1, l3in = regress(inLSTM1)
    gc.collect()

    '''
    feat cube contains the feature maps of the last convolution layer. of shape
    [[64L, 32L, 3L, 3L], [64L, 64L, 3L, 3L], [64L, 32L, 3L, 3L], [64L, 64L, 3L, 3L], 
    [64L, 64L, 3L, 3L],  [64L, 64L, 3L, 3L], [64L, 64L, 3L, 3L] ]
    '''

    inLSTM3 = torch.mul(l3in[0], feat_cube[2].view(1, 1, -1))
    regress.lstm3 = nn.LSTM(1024, 64*64, 1, bias=False,\
                            batch_first=False, dropout=0.3) #reshape last lstm layer

    if args.ship2gpu:
        regress.lstm3 = regress.lstm3.cuda() 
    gc.collect()

    for m in regress.modules():
        for t in m.state_dict().values():
            init.uniform(t, 0, 1)
    param3 = list(regress.parameters())
    for i in range(len(param3)):
        init.uniform(param3[i], 0, 1)
    y3, l2in = regress(inLSTM3)

    inLSTM2 = get_lstm_input(l2in, feat_cube[1])
    # Fix layers 1, 3, 4, 5, 6, 7  | layers 0 and 2 have unique shapes
    regress.lstm1 = nn.LSTM(64*64, 64*64, 1, bias=False, batch_first=False, dropout=0.3)
    regress.lstm2 = nn.LSTM(64*64, 64*16, 1, bias=False, batch_first=False, dropout=0.3)
    regress.lstm3 = nn.LSTM(64*16, 64*64, 1, bias=False, batch_first=False, dropout=0.3)

    if args.ship2gpu:
        regress = regress.cuda()

    #use normal initialization for regression layer
    for m in regress.modules():
        if(isinstance(m, nn.LSTM)):
            mvals = m.state_dict().values()
            init.uniform(mvals[0], 0, 1)
            init.uniform(mvals[1], 0, 1)
    params_n = list(regress.parameters())
    for i in range(len(params_n)):
        init.uniform(params_n[i], 0, 1)
    y2, l4in = regress(inLSTM2)

    inLSTM4 = get_lstm_input(l4in, feat_cube[3])
    y4, l5in = regress(inLSTM4)
    inLSTM5 = get_lstm_input(l5in, feat_cube[4])
    y5, l6in = regress(inLSTM5)
    inLSTM6 = get_lstm_input(l6in, feat_cube[5])
    y6, l7in = regress(inLSTM6)
    inLSTM7 = get_lstm_input(l7in, feat_cube[6])
    y7, l8in = regress(inLSTM7)

    # concatenate the attention variables
    attn_2 = y1
    attn_1 = torch.stack((y2, y3, y4, y5, y6, y7), 0).view(-1, y2.size(-1)).t()

    num_samples = 1
    replacement = True
    regress_input = torch.multinomial(attn_1, num_samples, replacement)

    global resnet
    resnet.saved_attention.append(regress_input)

    return regress_input

def rnn_stochastic_backprop(rlr):
    rewards = []
    global rnn_optimizer

    for r in resnet.saved_attention[::-1]:
        r = r.double()
        R, gamma = 0, 0.99
        R = r + gamma * R
        rewards.insert(0, R)

    rewards = rewards[0] 
    eps     = np.finfo(np.float64).eps
    rewards = (rewards - rewards.mean().data[0])/(rewards + rewards.std().data[0] + eps)
    rewards = rewards.cuda() if args.ship2gpu else rewards

    for action, r in zip(resnet.saved_attention, rewards):
        action.reinforce(action.data.double())

    rnn_optimizer.zero_grad()
    torch.autograd.backward(resnet.saved_attention, [None for _ in resnet.saved_attention])
    rnn_optimizer.step()

    del resnet.rewards[:]
    del resnet.saved_attention[:]

def trainClassifierRegressor(train_loader, bbox_loader, args):
    #cnn hyperparameters
    clr = args.clr
    batchSize = args.cbatchSize
    maxIter = args.cmaxIter

    #grab global vars
    global resnet, rnn_optimizer, regressor

    #rnn hyperparameters
    numLayers, regressLength = 2, 5
    noutputs, rlr = 12, args.rnnLR
    inputSize, nHidden = 128, [64, 32]

    #determine classification loss and clsfx_optimizer
    clsfx_crit   = nn.CrossEntropyLoss()
    regress_crit = nn.MSELoss()

    #define optimizer
    clsfx_optimizer = torch.optim.Adam(resnet.parameters(), clr)
    rnn_optimizer   = optim.Adam(regressor.parameters(), rlr)


    for k, v in enumerate(bbox_loader):
        targ_X  = v

    rewards, running_reward, targ_X = [], 10, targ_X[0]

    # Train classifier + regressor
    start_m = None
    for epoch in range(maxIter): #run through the images maxIter times
        for i, (train_X, train_Y) in enumerate(train_loader):

            if(args.ship2gpu):
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
                targ_X  = targ_X.cuda()
                resnet  = resnet.cuda()

            gc.collect()
            m = memory_usage()

            start_m = m if start_m is None else start_m            

            images, labels   = Variable(train_X), Variable(train_Y)

            #rnn input
            rtargets = Variable(targ_X[:,i:i+regressLength,:]).view(regressLength, -1)            
            regress_input = select_regress_input(resnet.layer3).double() #extract last convolution layer
            resnet.rewards.append(regress_input)

            X = len(regress_input)
            X_tr, X_te = int(0.8*X), int(0.2*X) 

            #train lstm regressor
            # print('regress_input[:X_tr]: ', regress_input)
            rtrain_X = regress_input[:X_tr].t().expand(regressLength, args.cbatchSize, X_tr)
            rtest_X = regress_input[X_tr:].t().expand(regressLength, args.cbatchSize, X_te+1)

            if(args.ship2gpu):
                rtrain_X    = rtrain_X.cuda()
                rtest_X     = rtest_X.cuda()
                regressor   = regressor.cuda()
                
            gc.collect()

            # Forward + Backward + Optimize
            clsfx_optimizer.zero_grad()
            rnn_optimizer.zero_grad()

            #predict classifier outs and regressor outputs
            outputs  = resnet(images)
            routputs = regressor(rtrain_X)

            gc.collect()

            # compute loss
            loss     = clsfx_crit(outputs, labels)      
            rloss    = regress_crit(routputs, rtargets)

            # backward pass
            loss.backward()     
            rnn_stochastic_backprop(rlr)

            # step optimizer
            clsfx_optimizer.step()   
            rnn_optimizer.step()    

            print ("Epoch [%d/%d], Iter [%d] cLoss: %.8f, rLoss: %.4f, mem_usage %.1f MB" %(epoch+1, maxIter, i+1,
                                                loss.data[0], rloss.data[0], m-start_m/256))

            if epoch % 5 == 0 and epoch >0:
                clr *= 1./epoch
                rlr *= 1./epoch

                clsfx_optimizer = optim.Adam(resnet.parameters(), clr)
                rnn_optimizer   = optim.Adam(regressor.parameters(), rlr)

    torch.save(regressor.state_dict(), 'regressnet_' + str(args.cmaxIter) + '.pkl')
    return resnet, regressor, rtest_X

def testClassifierRegressor(test_loader, resnet, regressnet, rtest_X, args):
    correct, total = 0, 0

    rtest_X = rtest_X.cuda() if args.ship2gpu else rtest_X
    for test_X, test_Y in test_loader:

        test_X = test_X.cuda() if args.ship2gpu else test_X
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
    torch.save(resnet.state_dict(), 'resnet_score='+ str(score) + '.pkl')

def main(args):
    #obtain training and testing data
    lnp = LoadAndParse(args)
    train_loader, test_loader, bbox_loader = lnp.partitionData()

    # train  conv+rnn nets
    # try:
    net, reg, rtest_X = trainClassifierRegressor(train_loader, bbox_loader, args)
    # test conv+rnn nets
    testClassifierRegressor(test_loader, net, reg, rtest_X, args)
    # except:
    #     print("stack overflow. please check the contents of the \'out.log\'' file")
    #     with open('out.log', 'w') as f:
    #         traceback.extract_stack()
    #         traceback.print_stack(file=f)
    #     pass


if __name__=='__main__':
    main(args)
