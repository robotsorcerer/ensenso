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

import sys, traceback
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

#myne utils
from model import ResNet, ResidualBlock, \
                StackRegressive, RecurrentModel
from utils import get_bounding_boxes as bbox

parser = argparse.ArgumentParser(description='Process environmental variables')
parser.add_argument('--ship2gpu', action='store_true', default=True, help="use cuda or not?")
parser.add_argument('--multinomial', action='store_true', default=False, help="sample from feature cube?")
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



# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = torch.randn(ndim, ndim)
    u, _, _ = torch.svd(W)
    return u

def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * torch.randn(nin, nout)
    return W

def pprint(x):
    print(str(x) + ' : ', x)

def trainClassifierRegressor(train_loader, bbox_loader, args):
    #cnn hyperparameters
    clr = args.clr
    batchSize = args.cbatchSize
    maxIter = args.cmaxIter

    #rnn hyperparameters
    numLayers, regressLength = 2, 5
    noutputs, rlr = 12, args.rnnLR
    inputSize, nHidden = 128, [64, 32]

    resnet = ResNet(ResidualBlock, [3, 3, 3])

    # #extract last convolution layer
    last_layer, feat_cube = resnet.layer3, []
    for param in last_layer.parameters():
        if param.dim() > 1:  # extract only conv cubes
            feat_cube.append(param)

    #max pool the last conv layer
    #define kernel size and width for pooled features
    kernel, stride = 3, 2
    #then apply a 2d max pooling of conv layer features
    m = nn.MaxPool2d(kernel, stride)
    feat_cube[0] = m(feat_cube[0]) #will now be 64 x 32 x 1 x 1
    feat_cube[2] = m(feat_cube[2]) #will now be 64 x 32 x 1 x 1
    feat_cube[1] = m(feat_cube[1])
    for x in xrange(3, 7):
        feat_cube[x] = m(feat_cube[x])

    lt = []  # this contains the soft max attention for each pooled layer
    for x in xrange(len(feat_cube)):
        temp = softmax(feat_cube[x])
        lt.append(temp)

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

    inLSTM1 = torch.mul(lt[0], feat_cube[0]).view(1,1,-1)  #will have 2048 connections
    regress = RecurrentModel(inputSize=inLSTM1.size(2), nHidden=[inLSTM1.size(2),1024, 64*32],\
                             noutputs=64*32,batchSize=args.cbatchSize, ship2gpu=args.ship2gpu, \
                             numLayers=1)
    #use normal initialization for regression layer
    params = list(regress.parameters())
    for i in range(len(params)):
        init.uniform(params[i], 0, 1)

    y1, l3in = regress(inLSTM1)

    '''
    feat cube contains the feature maps of the last convolution layer. of shape
    64L, 32L, 3L, 3L
    64L, 64L, 3L, 3L
    64L, 32L, 3L, 3L
    64L, 64L, 3L, 3L
    64L, 64L, 3L, 3L
    64L, 64L, 3L, 3L
    64L, 64L, 3L, 3L
    '''

    inLSTM3 = torch.mul(l3in[0], feat_cube[2].view(1, 1, -1))
    #reshape last lstm layer
    # regress = regress
    regress.lstm3 = nn.LSTM(1024, 64*64, 1, bias=False, batch_first=False, dropout=0.3)
    # for m in regress.modules():
    #     for t in m.state_dict().values():
    #         init.uniform(t, 0, 1)
    param3 = list(regress.parameters())
    # for i in range(len(param3)):
    #     init.uniform(param3[i], 0, 1)
    y3, l2in = regress(inLSTM3)

    inLSTM2 = torch.mul(l2in[0], feat_cube[1].view(1, 1, -1))
    # Fix layers 1, 3, 4, 5, 6, 7  | layers 0 and 2 have unique shapes
    regress.lstm1 = nn.LSTM(64*64, 64*64, 1, bias=False, batch_first=False, dropout=0.3)
    regress.lstm2 = nn.LSTM(64*64, 64*16, 1, bias=False, batch_first=False, dropout=0.3)
    regress.lstm3 = nn.LSTM(64*16, 64*64, 1, bias=False, batch_first=False, dropout=0.3)
    #use normal initialization for regression layer
    # for m in regress.modules():
    #     if(isinstance(m, nn.LSTM)):
    #         mvals = m.state_dict().values()
    #         init.uniform(mvals[0], 0, 1)
    #         init.uniform(mvals[1], 0, 1)
    params_n = list(regress.parameters())
    # for i in range(len(params_n)):
    #     init.uniform(params_n[i], 0, 1)
    y2, l4in = regress(inLSTM2)
    inLSTM4 = torch.mul(l4in[0], feat_cube[3].view(1, 1, -1))
    y4, l5in = regress(inLSTM4)
    inLSTM5 = torch.mul(l5in[0], feat_cube[4].view(1, 1, -1))
    y5, l6in = regress(inLSTM5)
    inLSTM6 = torch.mul(l6in[0], feat_cube[5].view(1, 1, -1))
    y6, l7in = regress(inLSTM6)
    inLSTM7 = torch.mul(l7in[0], feat_cube[6].view(1, 1, -1))
    y7, l8in = regress(inLSTM7)

    #zt_vec \in R^D captures the visual info associated with particular input locations
    zt_vec = [y1, y2, y3, y4, y5, y6, y7]

    # concatenate the attention variables
    attn_2 = torch.stack((zt_vec[1], zt_vec[2], zt_vec[3], zt_vec[4],\
                        zt_vec[5], zt_vec[6]), 0)
    attn_2 = attn_2.view(-1, attn_2.size(-1))
    attn_1 = zt_vec[0]

    """
    at run time we are going to draw samples from a multinomial
    probability distribution

    # regress_input will be a 1 x 4096 Tensor
    """

    # print(regress_input.size(), attn_2.size())
    if args.multinomial:
        regress_input = torch.multinomial(attn_2.t(), 1, True).double()
    else:
        sample_idx = np.random.randint(0, 7)
        regress_input = attn_2[sample_idx].unsqueeze(0).t()
        print('regress input: ', regress_input.size())

    #determine classification loss and clsfx_optimizer
    clsfx_crit = nn.CrossEntropyLoss()
    clsfx_optimizer = torch.optim.Adam(resnet.parameters(), clr)

    X_tr = int(0.8*len(regress_input))
    X_te = int(0.2*len(regress_input))
    X = len(regress_input)

    #reshape inputs
    rtrain_X = regress_input[:X_tr].t().expand(regressLength, X_tr)
    rtest_X = regress_input[X_tr:].t().expand(regressLength, X_te+1)

    print('rTrain_X size: ', rtrain_X.size())
    print('regress_input\': {}, rTest: {} '.format(regress_input[X_tr:].t().size(), rtest_X.size()))
    # Get regressor model and predict bounding boxes
    regressor = StackRegressive(inputSize=rtrain_X.size(1), nHidden=[rtrain_X.size(1),rtrain_X.size(1)/4, 8], noutputs=8, batchSize=args.cbatchSize, ship2gpu=args.ship2gpu, numLayers=1)

    targ_X = None
    for _, targ_X in bbox_loader:
        targ_X = targ_X

    # time.sleep(50)
    if(args.ship2gpu):
        rtrain_X = rtrain_X.cuda()
        rtest_X  = rtest_X.cuda()
        targ_X = targ_X.cuda()

    #define optimizer
    rnn_optimizer = optim.SGD(regressor.parameters(), rlr)

    # Train classifier
    for epoch in range(maxIter): #run through the images maxIter times
        for i, (train_X, train_Y) in enumerate(train_loader):

            if(args.ship2gpu):
                train_X = train_X.cuda()
                train_Y = train_Y.cuda()
                resnet  = resnet.cuda()

            images = Variable(train_X)
            labels = Variable(train_Y)
            #rnn input
            rtargets = targ_X[:,i:i+regressLength,:]
            #reshape targets for inputs
            rtargets = Variable(rtargets.view(regressLength, -1))

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
        # routputs = regressnet(rtest_X)

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
    try:
        net, reg, rtest_X = \
                    trainClassifierRegressor(train_loader, bbox_loader, args)
        
        # test conv+rnn nets
        testClassifierRegressor(test_loader, net, reg, rtest_X, args)
    except:
        print("stack overflow. please check the contents of the \'out.log\'' file")
        with open('out.log', 'w') as f:
            traceback.extract_stack()
            traceback.print_stack(file=f)


if __name__=='__main__':
    main(args)
