import cv2
import functools
import json
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class TestUrbanClassDataset(Dataset):
	def __init__(self, path, direction):
		assert direction in ['a2b',
		                     'b2a'], 'Unknown direction {}, expected a2b or b2a.'.format(
			direction)
		super(TestUrbanClassDataset, self).__init__()
		self.direction = direction
		self.path = path
		self.filenames = [x for x in os.listdir(self.path)]
		# with open('gan_maps/{}/labels_density.json'.format(city)) as f:
		#   self.json = json.load(f)

		transform_list = [transforms.ToTensor(),
		                  transforms.Normalize((0.5, 0.5, 0.5),
		                                       (0.5, 0.5, 0.5))]

		self.transform = transforms.Compose(transform_list)

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()
		a, b = self._get_images(index)
		a = transforms.ToTensor()(a)
		b = transforms.ToTensor()(b)

		a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
		b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
		label = self.filenames[index]

		if self.direction == "a2b":
			return a, b, label
		else:
			return b, a, label

	def __len__(self):
		return len(self.filenames)

	def _get_images(self, index):
		image = cv2.imread(self.path + '/' + self.filenames[index])
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		a = image[:, :int(image.shape[1] / 2)]
		b = image[:, int(image.shape[1] / 2):]
		return a, b

	def show(self, index):
		a, b = self._get_images(index)
		plt.subplot(1, 2, 1)
		plt.imshow(a)
		plt.subplot(1, 2, 2)
		plt.imshow(b)