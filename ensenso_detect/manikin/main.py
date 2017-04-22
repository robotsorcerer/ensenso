#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from PIL import Image
from os import listdir
import json
import time
import numpy as np

from random import shuffle

from model import ResNet, ResidualBlock

import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

class loadAndParse():

	def __init__(self, args, true_path="raw/true/", fake_path="raw/fake/"):
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
		self.fake_path= fake_path

		#define tensors to hold the images in memory
		self.fakenreal_images, self.fakenreal_labels  = [], []


	# #load labels file
	def loadLabelsFromJson(self):
		labels_file = open('labels.json').read()
		labels = json.loads(labels_file)
		classes = [ labels["0"], labels["1"]]  # 0 = fake, 1=real
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
		true_labels = [1]*len(true_images)  #faces
		fake_labels = [0]*len(fake_images)  #backgrounds

		imagesAll = true_images + fake_images
		labelsAll = true_labels + fake_labels

		#permute images and labels in place
		shuffle(imagesAll)
		# imagesAll = imagesAll[torch.randperm(len(imagesAll))]
		shuffle(labelsAll)
		#
		# for x in range(len(imagesAll)):
		# 	# print(imagesAll[x].size(0))
		# 	print(imagesAll[x].size())
		# 	if x == 5:
		# 		break

		classes = self.loadLabelsFromJson()

		#be sure the images are rightly loaded
		if self.args.disp:
			true_images[0].show()
			fake_images[0].show()

		# Now preprocess and create list for images
		for imgs in imagesAll:
			images_temp = self.preprocess(imgs)
			# print("size of image, {}, {}", imgs, images_temp.size())
			'''
			For some reason, not cropping the images makes the dimensions inconsistent
			Make sure the images are cropped before preprocessing
			We will not include those in our training and testing images
			'''
			if images_temp.size(0) == 3:
				self.fakenreal_images.append(images_temp)
			# print("size: {} ".format(images_temp.size(0)))

		# for x in range(len(self.fakenreal_images)):
		# 	print('fake_images size: ', self.fakenreal_images[x].size())

		self.fakenreal_labels = labelsAll

		if self.args.disp:
			print(fakenreal_images[240])

		if self.args.verbose:
			# #be sure the images are properly loaded in memory
			print("\nTotal # of AllTensors: {}, images size: {}".format(len(self.fakenreal_images),
										self.fakenreal_images[64].size()))

	def partitionData(self):
		# retrieve the images first
		self.getImages()

		#by now the self.fakenreal_images and fakenreak_labels lists are populated
#
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
		# print('\ntrain_X {}, train_Y: {}, X_te: {} \n'.format(train_X.size(), train_Y.size(), X_te))

		#testing set
		test_X = torch.stack(self.fakenreal_images[X_tr:], 0)
		test_Y = torch.from_numpy(np.array(self.fakenreal_labels[X_tr+1:]))

		#check size of slices
		if self.args.verbose:
			print('train_X and train_Y sizes: {} | {}'.format(train_X.size(), train_Y.size()))
			print('test_X and test_Y sizes: {} | {}'.format(test_X.size(), test_Y.size()))

		return train_X, train_Y, test_X, test_Y

def main():
	parser = argparse.ArgumentParser(description='Process environmental variables')
	parser.add_argument('--verbose', type=bool, default=False,
						help='print out shit')
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--disp', type=bool, default=False)
	parser.add_argument('--maxIter', type=int, default=50)
	parser.add_argument('--num_iter', type=int, default=5)
	parser.add_argument('--lr', type=float, default=1e-3)
	args = parser.parse_args()

	#obtain training and testing data
	lnp = loadAndParse(args)
	train_X, train_Y, test_X, test_Y = lnp.partitionData()

	#obtain model
	resnet = ResNet(ResidualBlock, [3, 3, 3])
	if(args.cuda):
		resnet = resnet.cuda()
		train_X = train_X.cuda()
		train_Y = train_Y.cuda()
		test_X = test_X.cuda()
		test_Y = test_Y.cuda()

	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	lr = args.lr
	maxIter = args.maxIter
	optimizer = torch.optim.Adam(resnet.parameters(), lr=args.lr)

	# Training
	for epoch in range(maxIter):
		for i in range(args.num_iter): #loop 5 times on each bach per epoch
			images = Variable(train_X)
			labels = Variable(train_Y)

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			outputs = resnet(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" %(epoch+1, args.maxIter, i+1, args.num_iter, loss.data[0]))
			# print('full loss: ' , loss.data)

			# Decaying Learning Rate
			if (epoch) % 10 == 0:
				lr /= 3
				optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

	# Test
	correct = 0
	total = 0
	for epoch in range(epoch):
		images = Variable(test_X)
		labels = Variable(test_Y)
		outputs = resnet(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted.cpu() == labels).sum()

	print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

	# Save the Model
	torch.save(resnet.state_dict(), 'resnet.pkl')

if __name__=='__main__':
	main()
