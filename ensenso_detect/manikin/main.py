#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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

from random import shuffle

from model import ResNet, ResidualBlock

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)


torch.set_default_tensor_type('torch.DoubleTensor')

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
		temp = list(zip(imagesAll, labelsAll))
		shuffle(temp)
		imagesAll, labelsAll = zip(*temp)

		classes = self.loadLabelsFromJson()

		#be sure the images are rightly loaded
		if self.args.disp:
			true_images[0].show()
			fake_images[0].show()

		# Now preprocess and create list for images
		for imgs in imagesAll:
			'''
			For some reason, not cropping the images makes the dimensions inconsistent
			Make sure the images are cropped before preprocessing
			We will not include those in our training and testing images
			'''
			images_temp = self.preprocess(imgs).double()
			if images_temp.size(0) == 3:
				self.fakenreal_images.append(images_temp)
		self.fakenreal_labels = labelsAll

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
		train_dataset =  data.TensorDataset(train_X, train_Y)
		train_loader = data.DataLoader(train_dataset,
						batch_size=self.args.batchSize, shuffle=True)

		#testing set
		test_X = torch.stack(self.fakenreal_images[X_tr:], 0)
		test_Y = torch.from_numpy(np.array(self.fakenreal_labels[X_tr+1:]))
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
	parser = argparse.ArgumentParser(description='Process environmental variables')
	parser.add_argument('--verbose', type=bool, default=False,
						help='print out shit')
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--disp', type=bool, default=False)
	parser.add_argument('--maxIter', type=int, default=1000)
	parser.add_argument('--num_iter', type=int, default=5)
	parser.add_argument('--batchSize', type=int, default=20)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--epoch', type=int, default=500)
	args = parser.parse_args()

	#obtain training and testing data
	lnp = loadAndParse(args)
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

			print ("Epoch [%d/%d], Iter [%d/%d] Loss: %.8f" %(epoch+1, args.maxIter, i+1, int(batchSize/2), loss.data[0]))

			# Decaying Learning Rate
			if (epoch+1) % 50 == 0:
				lr /= 3
				optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)

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

	print('Accuracy of the model on the test images: %d %%' %(100 * correct / total))

	# Save the Model
	torch.save(resnet.state_dict(), 'resnet.pkl')

if __name__=='__main__':
	main()
