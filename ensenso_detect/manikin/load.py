#!/usr/bin/env python3

import torch
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

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
	 color_scheme='Linux', call_pdb=1)

torch.set_default_tensor_type('torch.DoubleTensor')

class loadAndParse():

	def __init__(self, args, true_path="raw/fake/"):

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

		#define labels
		self.real_labels = [1]*len(true_images)  #faces

		classes = self.loadLabelsFromJson()

		#be sure the images are rightly loaded
		if self.args.disp:
			true_images[0].show()

		# Now preprocess and create list for images
		for imgs in true_images:
			# cast to double since preprocess sends to FloatTensor by default
			images_temp = self.preprocess(imgs).double()

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
		loader   = data.DataLoader(dataset, batch_size=self.args.batchSize, shuffle=True)

		return loader

def main():
	parser = argparse.ArgumentParser(description='Process environmental variables')
	parser.add_argument('--verbose', type=bool, default=False)
	parser.add_argument('--epoch', type=int, default=500)
	parser.add_argument('--disp', type=bool, default=False)
	parser.add_argument('--cuda', type=bool, default=True)
	parser.add_argument('--batchSize', type=int, default=1)
	parser.add_argument('--model', type=str, default='resnet_acc=80_iter=200.pth')
	args = parser.parse_args()

	lnp = loadAndParse(args)

	classes = lnp.loadLabelsFromJson()
	loader  = lnp.getImagesAsTensors()
#
	model = torch.load('models225/' + args.model)
	model.eval()

	if not args.cuda:
		model.cpu()

	if(args.verbose):
		print(model)

	#define PIL primitives
	to_pil = transforms.ToPILImage()

	for images, labels in loader:
		output = model(Variable(images.cuda()))
		_, predicted = torch.max(output, 1)

		#collect classes
		classified = predicted.data[0][0]
		index = int(classified)
		img_class = classes[str(index)]

		#display image and class
		# img = to_pil(images)
		print('class of image', classes[str(index)])

if __name__ == '__main__':
	main()
