#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
from PIL import Image
from os import listdir
import json
import time
import numpy as np
import os
# import setGPU

from collections import namedtuple
import numpy as np
import cv2

from random import shuffle

from model import ResNet, ResidualBlock
from utils import get_bounding_boxes as bbox

import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

parser = argparse.ArgumentParser(description='Process environmental variables')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--disp', type=bool, default=False)
parser.add_argument('--maxIter', type=int, default=200)
parser.add_argument('--num_iter', type=int, default=5)
parser.add_argument('--batchSize', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

torch.set_default_tensor_type('torch.DoubleTensor')

class loadAndParse(object):

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

    def load_negative(self):
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
        First
        Followed 4 cols represent top coordinates of face boxes,
        Followed by lower coordinates of face_boxes
        Next 2 cols belong to left eye centers, last col are right eye coords
        """
        self.face_left_right = np.concatenate((faces_top, faces_bot, left_right), axis=1)

        #define labels
        self.real_labels = [1]*len(self.true_images)  #faces
        # self.fake_labels = [0]*len(fake_images)  #backgrounds

        imagesAll = self.true_images #+ fake_images

        """
        labelsAll = true_labels #+ fake_labels

        #permute images and labels in place
        temp = list(zip(imagesAll, labelsAll))
        shuffle(temp)
        imagesAll, labelsAll = zip(*temp)

        classes = self.loadLabelsFromJson()

        #be sure the images are rightly loaded
        if self.args.disp:
            true_images[0].show()
            fake_images[0].show()
        """

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

        #by now the self.real_images and fakenreak_labels lists are populated
#
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
        # Now labels and bboxes
        labels_t = torch.from_numpy(np.array(self.real_labels))
        boxes_t = torch.from_numpy(np.array(self.face_left_right))

        #Allocate storage for all labels
        labelsAll = torch.LongTensor(labels_t.size(0), boxes_t.size(1)+1)

        labelsAll[:,0:1] = labels_t
        labelsAll[:,1:] = boxes_t

        labelsAll = labelsAll

        #Now copy tensors over
        train_X = torch.stack(self.real_images[:X_tr], 0)
        train_Y = labelsAll[:X_tr,:].double()
        #testing set
        test_X = torch.stack(self.real_images[X_tr:], 0)
        test_Y = labelsAll[X_tr:].double()

        #data loaders
        train_dataset =  data.TensorDataset(train_X, train_Y)
        train_loader = data.DataLoader(train_dataset,
                        batch_size=self.args.batchSize, shuffle=True)


        test_dataset = data.TensorDataset(test_X, test_Y)
        test_loader = data.DataLoader(test_dataset,
                            batch_size=self.args.batchSize, shuffle=True)

        #check size of slices
        if self.args.verbose:
            print('train_X and train_Y sizes: {} | {}'.format(train_X.size(), train_Y.size()))
            print('test_X and test_Y sizes: {} | {}'.format(test_X.size(), test_Y.size()))

        return train_loader, test_loader

    def file_exists(file_path):
        if not file_path:
            return False
        else:
            return True

def main():
    #obtain training and testing data
    lnp = loadAndParse(args)

    dict_combo = bbox()
    faces_dict, left_dict, right_dict = dict_combo[0], dict_combo[1],  dict_combo[2]

    if args.verbose:
        print(faces_dict)
        # print(left_dict)
        # print(right_dict)

    train_loader, test_loader = lnp.partitionData()

    #obtain model
    resnet = ResNet(ResidualBlock, [3, 3, 3])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    batchSize = args.batchSize
    maxIter = args.maxIter
    optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)

    # Training
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
            optimizer.zero_grad()
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if(epoch %2 == 0):
            print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.8f" %(epoch+1, args.maxIter, i+1, int(batchSize), loss.data[0]))

            # Decaying Learning Rate
            # if (epoch+1) % 50 == 0:
            #     lr /= 3
            #     optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

    # Test
    correct = 0
    total = 0
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
    torch.save(resnet.state_dict(), 'resnet_'+ score + '%_epoch_' + args.maxIter)
    # torch.save(resnet, 'resnet.pth')

if __name__=='__main__':
    main()
