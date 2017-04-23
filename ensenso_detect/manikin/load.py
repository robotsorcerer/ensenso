#!/usr/bin/env python3

import torch
from os import listdir
from PIL import Image
from torch.autograd import Variable
import json
import argparse
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from random import shuffle

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
	 color_scheme='Linux', call_pdb=1)


class loadAndParse():

	def __init__(self, args, true_path="raw/true/"):

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

		#define tensors to hold the images in memory
		self.real_images, self.real_labels = [], []

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

		#define labels
		self.real_labels = [1]*len(true_images)  #faces

		classes = self.loadLabelsFromJson()

		#be sure the images are rightly loaded
		if self.args.disp:
			true_images[0].show()

		# Now preprocess and create list for images
		for imgs in true_images:
			images_temp = self.preprocess(imgs)

			if images_temp.size(0) == 3:
				self.real_images.append(images_temp)

		if self.args.disp:
			print(real_images[240])

		if self.args.verbose:
			# #be sure the images are properly loaded in memory
			print("\nTotal # of AllTensors: {}, images size: {}".format(len(self.real_images),
										self.real_images[64].size()))

	def getImagesAsTensors(self):

		self.getImages()

		X_len = len(self.real_images)

		X = torch.LongTensor(X_len, self.real_images[0].size(0), self.real_images[0].size(1),
								  self.real_images[0].size(2))

		X_tensors = torch.stack(self.real_images[:], 0)
		Y_tensors = torch.from_numpy(np.array(self.real_labels[:]))

		dataset = data.TensorDataset(X_tensors, Y_tensors)
		loader   = data.DataLoader(dataset, batch_size=20, shuffle=True)

		return loader

def main():
	parser = argparse.ArgumentParser(description='Process environmental variables')
	parser.add_argument('--verbose', type=bool, default=False)
	parser.add_argument('--epoch', type=int, default=500)
	parser.add_argument('--disp', type=bool, default=False)
	args = parser.parse_args()

	lnp = loadAndParse(args)
	loader = lnp.getImagesAsTensors()

	model = torch.load('models225/resnet_acc=97_iter=1000.pkl')

	if(args.verbose):
		print(model)

	for images, labels in loader:
		output = model(Variable(images))
		print('outputs ')

if __name__ == '__main__':
    main()
