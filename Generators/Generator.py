import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.custom_layers import EqualizedConv2d, NormalizationLayer, EqualizedLinear
import numpy as np
#in,out,kernel_size,stride,padding
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.random_vector_size = 50
		self.no_of_upscales = 3
		out_channels_final = 1
		self.upscale_list = nn.ModuleList([])
		self.to_rgb_list = nn.ModuleList([])

		self.dense = EqualizedLinear(self.random_vector_size, 4*4*50)

		self.conv1 = EqualizedConv2d(50,50,3,1)
		self.to_rgb_4x4 =  EqualizedConv2d(50,out_channels_final,1,0)

		self.conv_1 = EqualizedConv2d(50,50,3,1)
		self.conv_2 = EqualizedConv2d(50,50,3,1)
		self.to_rgb1 = EqualizedConv2d(50,out_channels_final,1,0)
		self.conv_3 = EqualizedConv2d(50,50,3,1)
		self.conv_4 = EqualizedConv2d(50,50,3,1)
		self.to_rgb2 = EqualizedConv2d(50,out_channels_final,1,0)
		self.conv_5 = EqualizedConv2d(50,25,3,1)
		self.conv_6 = EqualizedConv2d(25,25,3,1)
		self.to_rgb3 = EqualizedConv2d(25,out_channels_final,1,0)


		self.norm_layer = NormalizationLayer()
		self.lr   = nn.LeakyReLU(negative_slope=0.2)

	def forward(self,batch_size):
		# batch_size = len(one_hot)
		out_images_list = []
		z = torch.randn(batch_size, self.random_vector_size).cuda()
		# z = torch.cat( (z,one_hot),1)
		x = self.dense(z)
		x = torch.reshape(x,(batch_size,50,4,4))

		x = self.lr(self.norm_layer(self.conv1(x)))

		image_4x4 = self.to_rgb_4x4(x)
		out_images_list.append(image_4x4)

		x = F.interpolate(x,scale_factor=2,mode='nearest')
		x = self.lr(self.norm_layer(self.conv_1(x)))
		x = self.lr(self.norm_layer(self.conv_2(x)))
		image = self.to_rgb1(x)
		out_images_list.append(image)
		x = F.interpolate(x,scale_factor=2,mode='nearest')
		x = self.lr(self.norm_layer(self.conv_3(x)))
		x = self.lr(self.norm_layer(self.conv_4(x)))
		image2 = self.to_rgb2(x)
		out_images_list.append(image2)
		x = F.interpolate(x,scale_factor=2,mode='nearest')
		x = self.lr(self.norm_layer(self.conv_5(x)))
		x = self.lr(self.norm_layer(self.conv_6(x)))
		image3 = self.to_rgb3(x)
		out_images_list.append(image3)

		return out_images_list










