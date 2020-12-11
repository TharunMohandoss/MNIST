import random
import cv2
import numpy as np
import time
class Augmentation:
	def __init__(self,config):
		self.config = config

	def apply_mirror_augmentation(self,data_batch):
		random_number = random.randint(1,2)
		if random_number == 1:
			for data_type in data_batch:
				data_batch[data_type] = data_batch[data_type][:,:,:,::-1]
		return data_batch


	def random_resize(self,data_batch):
		# print('initla batch size : ',len(data_batch['images']))
		random_scale = random.uniform(0.5,1.5)
		# print('scale : ',random_scale)
		new_height = data_batch['images'].shape[2]*random_scale
		new_width = data_batch['images'].shape[3]*random_scale
		dict_batch = dict()
		for data_type in data_batch:
			dict_batch[data_type] = []
		for data_type in data_batch:
			interpolation = cv2.INTER_NEAREST
			if data_type == "images":
				interpolation = cv2.INTER_CUBIC
			for i in range(int(len(data_batch[data_type])/max(random_scale*random_scale,1))):
				# print('type : ',data_type)
				intial_image = np.transpose(data_batch[data_type][i],(1,2,0))
				# print('intiial image shape : ',intial_image.shape)
				# print('initial image : ',intial_image[0,0])
				# print('interpolation : ',interpolation)
				if data_type == "images":
					intial_image*= 255
				resized_image = cv2.resize(intial_image,(int(new_width),int(new_height)),interpolation=interpolation)
				if (len(resized_image.shape)==2):
					resized_image = np.expand_dims(resized_image,axis=2)
				# print('shape of resized : ',resized_image.shape)
				# print('resized : ',resized_image[0,0])
				resized_image = np.transpose(resized_image,(2,0,1))
				if data_type == "images":
					resized_image /= 255.0
				dict_batch[data_type].append(resized_image)
		for data_type in data_batch:
			dict_batch[data_type] = np.asarray(dict_batch[data_type])
		# print('new batch size : ',len(dict_batch["images"]))
		# if len(dict_batch["images"]==0):
		# 	return data_batch
		return dict_batch


	# def apply_random_scale(self,data_batch):
	# 	ratio = random.uniform(0.5, 1.5)
 #        w, h = image.size
 #        tw = int(ratio * w)
 #        th = int(ratio * h)
 #        if ratio == 1:
 #            return image, label
 #        elif ratio < 1:
 #            interpolation = Image.ANTIALIAS
 #        else:
 #            interpolation = Image.CUBIC

 #        resized_images = []
 #        for image in data_batch["images"]:

	def augment_batch(self,data_batch):
	
		# data_batch = self.apply_mirror_augmentation(data_batch)
		# extra_aug_time = time.time()
		# data_batch = self.random_resize(data_batch)
		# print('random resize : ',time.time()-extra_aug_time)
		return data_batch