#!/usr/bin/env python

import torch
import os
from os import listdir
from PIL import Image
from PIL import ImageFont, ImageDraw
from torch.autograd import Variable
import json
import argparse
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from random import shuffle

import torch.nn as nn
from torchvision import models
import cv2

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
	 color_scheme='Linux', call_pdb=1)

from model import ResNet, ResidualBlock
from matplotlib import pyplot as plt

torch.set_default_tensor_type('torch.DoubleTensor')

#class to get values from multiple layers with one forward pass
class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.conv2 = nn.Conv2d(1, 1, 3)
        self.conv3 = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = F.relu(self.conv3(out2))
        return out1, out2, out3

# fetch the intermediate values when the forward
# behavior is defined by nn.Sequential()
class SelectiveSequential(nn.Module):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for key, module in modules_dict.items():
            self.add_module(key, module)
        self._to_select = to_select

    def forward(x):
        lister = []
        for name, module in self._modules.iteritems():
            x = module(x)
            if name in self._to_select:
                lister.append(x)
        return lister

# use selective sequential like this
class Net_SelectiveSequential(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = SelectiveSequential(
            ['conv1', 'conv3'],
            {'conv1': nn.Conv2d(1, 1, 3),
             'conv2': nn.Conv2d(1, 1, 3),
             'conv3': nn.Conv2d(1, 1, 3)}
        )

    def forward(self, x):
        return self.features(x)


class loadAndParse():

	def __init__(self, args, true_path="raw/face_pos/", fake_path="raw/face_pos/"):

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

		#provide path to true images
		self.true_path= true_path
		self.fake_path= fake_path

		#define tensors to hold the images in memory
		self.real_images, self.real_labels = [], []
		self.fake_images, self.fake_labels = [], []

	# #load labels file
	def loadLabelsFromJson(self):
		labels_file = open('labels.json').read()
		labels = json.loads(labels_file)
		classes = labels  # 0 = fake, 1=real
		return classes

	def loadImages(self, path):
		# return array of images

		imagesList = listdir(path)
		loadedImages = []
		for image in imagesList:
			img = Image.open(path + image)
			loadedImages.append(img)

		return loadedImages

	# get images in the dir
	def getImages(self):
		#load images
		true_images = self.loadImages(self.true_path)
		fake_images = self.loadImages(self.fake_path)

		#define labels
		self.real_labels = [1]*len(true_images)  #faces
		self.fake_labels = [0]*len(fake_images)

		classes = self.loadLabelsFromJson()

		#be sure the images are rightly loaded
		if self.args.disp:
			true_images[0].show()
			fake_images[0].show()

		# Now preprocess and create list for images
		for imgs in true_images:
			# cast to double since preprocess sends to FloatTensor by default
			images_temp = self.preprocess(imgs).double()
			if images_temp.size(0) == 3:
				self.real_images.append(images_temp)

		for imgs in fake_images:
			# cast to double since preprocess sends to FloatTensor by default
			images_temp = self.preprocess(imgs).double()
			if images_temp.size(0) == 3:
				self.fake_images.append(images_temp)

		if self.args.disp:
			print(self.real_images[3])
			print(self.fake_images[2])

		if self.args.verbose:
			# #be sure the images are properly loaded in memory
			print("\nTotal # of AllTensors: {}, images size: {}".format(len(self.real_images),
										self.real_images[64].size()))

	def getImagesAsTensors(self):

		self.getImages()

		Xtr_len = len(self.real_images)
		Xfk_len = len(self.fake_images)

		Xtr_tensors = torch.LongTensor(Xtr_len, self.real_images[0].size(0), self.real_images[0].size(1),
								  self.real_images[0].size(2))
		Xfk_tensors = torch.LongTensor(Xfk_len, self.fake_images[0].size(0), self.fake_images[0].size(1),
								  self.fake_images[0].size(2))

		Xtr_tensors = torch.stack(self.real_images[:], 0)
		Ytr_tensors = torch.from_numpy(np.array(self.real_labels[:]))

		Xfk_tensors = torch.stack(self.fake_images[:], 0)
		Yte_tensors = torch.from_numpy(np.array(self.fake_labels[:]))

		tr_dataset = data.TensorDataset(Xtr_tensors, Ytr_tensors)
		tr_loader   = data.DataLoader(tr_dataset, batch_size=self.args.batchSize, shuffle=True)

		return tr_loader, Xfk_tensors

def main():
	parser = argparse.ArgumentParser(description='Process environmental variables')
	parser.add_argument('--feature', dest='feature', action='store_true')
	parser.add_argument('--no-feature', dest='feature', action='store_false')
	parser.set_defaults(feature=True)
	parser.add_argument('--verbose', type=bool, default=False)
	parser.add_argument('--epoch', type=int, default=500)
	parser.add_argument('--disp', type=bool, default=False)
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--pkl_model', type=int, default=1)
	parser.add_argument('--fake_test', type=int, default=0)
	parser.add_argument('--batchSize', type=int, default=1)
	parser.add_argument('--model', type=str, default='resnet_acc=97_iter=1000.pkl')
	args = parser.parse_args()

	lnp = loadAndParse(args)
	classes = lnp.loadLabelsFromJson()
	tr_loader, test_X  = lnp.getImagesAsTensors()

	base, ext = os.path.splitext(args.model)
	if (ext == ".pkl"):  #using high score model
		model = ResNet(ResidualBlock, [3, 3, 3]).cuda()
		model.load_state_dict(torch.load('models225/' + args.model))
		# print(model.load_state_dict(torch.load('models225/' + args.model)))
	else:
		model = torch.load('models225/' + args.model)
		model.eval()

	if not args.cuda:
		model.cpu()

	#get last layer from resnet
	last_layer = nn.Sequential(*list(model.children()))[:-1]
	model.classifier = last_layer

	print(last_layer)

	'''
	remove last fully connected layer
	this will contain the features extracted by the convnet
	'''
	# eye_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
	# model.classifier = eye_classifier
	print('using model: ', args.model)

	corrIdx, Idx = 0, 0
	if (args.fake_test==1):
		for i in range(test_X.size(0)):
			output = model(Variable(test_X.cuda()))
			_, predicted = torch.max(output, 1)

			#collect classes
			classified = predicted.data[0][0]
			index = int(classified)

			if index == 0:  #fake
				corrIdx += 1
			Idx += 1

			img_class = classes[str(index)]

			#display image and class
			print('class \'o\' image', classes[str(index)])


		print('\n\ncorrectly classified: %d %%' %(100* corrIdx / Idx) )

	else:
		for images, labels in tr_loader:
			output = model(Variable(images.cuda()))
			_, predicted = torch.max(output, 1)

			#collect classes
			classified = predicted.data[0][0]
			index = int(classified)

			if index == 1:  #real
				corrIdx += 1
			Idx += 1
			img_class = classes[str(index)]

			#display image and class
			# print('class of image', classes[str(index)])

		print('\n\ncorrectly classified: %d %%' %(100* corrIdx / Idx) )


if __name__ == '__main__':
	main()
