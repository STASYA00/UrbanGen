import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from utils import *


class ContractingBlock(nn.Module):
	'''
	ContractingBlock Class
	Performs two convolutions followed by a max pool operation.
	Values:
		input_channels: the number of channels to expect from a given input
	'''
	def __init__(self, input_channels, use_dropout=False, use_bn=True):
		super(ContractingBlock, self).__init__()
		self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
		self.activation = nn.LeakyReLU(0.2)
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		if use_bn:
			self.batchnorm = nn.BatchNorm2d(input_channels * 2)
		self.use_bn = use_bn
		if use_dropout:
			self.dropout = nn.Dropout()
		self.use_dropout = use_dropout

	def forward(self, x):
		'''
		Function for completing a forward pass of ContractingBlock:
		Given an image tensor, completes a contracting block and returns the transformed tensor.
		Parameters:
			x: image tensor of shape (batch size, channels, height, width)
		'''
		x = self.conv1(x)
		if self.use_bn:
			x = self.batchnorm(x)
		if self.use_dropout:
			x = self.dropout(x)
		x = self.activation(x)
		x = self.conv2(x)
		if self.use_bn:
			x = self.batchnorm(x)
		if self.use_dropout:
			x = self.dropout(x)
		x = self.activation(x)
		x = self.maxpool(x)
		return x


class ExpandingBlock(nn.Module):
	'''
	ExpandingBlock Class:
	Performs an upsampling, a convolution, a concatenation of its two inputs,
	followed by two more convolutions with optional dropout
	Values:
		input_channels: the number of channels to expect from a given input
	'''
	def __init__(self, input_channels, use_dropout=False, use_bn=True):
		super(ExpandingBlock, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
		self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
		if use_bn:
			self.batchnorm = nn.BatchNorm2d(input_channels // 2)
		self.use_bn = use_bn
		self.activation = nn.ReLU()
		if use_dropout:
			self.dropout = nn.Dropout()
		self.use_dropout = use_dropout

	def forward(self, x, skip_con_x):
		'''
		Function for completing a forward pass of ExpandingBlock:
		Given an image tensor, completes an expanding block and returns the transformed tensor.
		Parameters:
			x: image tensor of shape (batch size, channels, height, width)
			skip_con_x: the image tensor from the contracting path (from the opposing block of x)
					for the skip connection
		'''
		x = self.upsample(x)
		x = self.conv1(x)
		skip_con_x = crop(skip_con_x, x.shape)
		x = torch.cat([x, skip_con_x], axis=1)
		x = self.conv2(x)
		if self.use_bn:
			x = self.batchnorm(x)
		if self.use_dropout:
			x = self.dropout(x)
		x = self.activation(x)
		x = self.conv3(x)
		if self.use_bn:
			x = self.batchnorm(x)
		if self.use_dropout:
			x = self.dropout(x)
		x = self.activation(x)
		return x


class FeatureMapBlock(nn.Module):
	'''
	FeatureMapBlock Class
	The final layer of a U-Net -
	maps each pixel to a pixel with the correct number of output dimensions
	using a 1x1 convolution.
	Values:
		input_channels: the number of channels to expect from a given input
		output_channels: the number of channels to expect for a given output
	'''
	def __init__(self, input_channels, output_channels):
		super(FeatureMapBlock, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

	def forward(self, x):
		'''
		Function for completing a forward pass of FeatureMapBlock:
		Given an image tensor, returns it mapped to the desired number of channels.
		Parameters:
			x: image tensor of shape (batch size, channels, height, width)
		'''
		x = self.conv(x)
		return x


class UNet(nn.Module):
	'''
	UNet Class
	A series of 4 contracting blocks followed by 4 expanding blocks to
	transform an input image into the corresponding paired image, with an upfeature
	layer at the start and a downfeature layer at the end.
	Values:
		input_channels: the number of channels to expect from a given input
		output_channels: the number of channels to expect for a given output
	'''
	def __init__(self, input_channels, output_channels, hidden_channels=32):
		super(UNet, self).__init__()
		self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
		self.contract1 = ContractingBlock(hidden_channels, use_dropout=True)
		self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
		self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
		self.contract4 = ContractingBlock(hidden_channels * 8)
		self.contract5 = ContractingBlock(hidden_channels * 16)
		self.contract6 = ContractingBlock(hidden_channels * 32)
		self.expand0 = ExpandingBlock(hidden_channels * 64)
		self.expand1 = ExpandingBlock(hidden_channels * 32)
		self.expand2 = ExpandingBlock(hidden_channels * 16)
		self.expand3 = ExpandingBlock(hidden_channels * 8)
		self.expand4 = ExpandingBlock(hidden_channels * 4)
		self.expand5 = ExpandingBlock(hidden_channels * 2)
		self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		'''
		Function for completing a forward pass of UNet:
		Given an image tensor, passes it through U-Net and returns the output.
		Parameters:
			x: image tensor of shape (batch size, channels, height, width)
		'''
		x0 = self.upfeature(x)
		x1 = self.contract1(x0)
		x2 = self.contract2(x1)
		x3 = self.contract3(x2)
		x4 = self.contract4(x3)
		x5 = self.contract5(x4)
		x6 = self.contract6(x5)
		x7 = self.expand0(x6, x5)
		x8 = self.expand1(x7, x4)
		x9 = self.expand2(x8, x3)
		x10 = self.expand3(x9, x2)
		x11 = self.expand4(x10, x1)
		x12 = self.expand5(x11, x0)
		xn = self.downfeature(x12)
		return self.sigmoid(xn)