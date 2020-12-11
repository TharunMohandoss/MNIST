from mnist import MNIST
import numpy as np
import cv2
# mndata = MNIST('./samples')

# images, labels = mndata.load_training()
# print(np.asarray(images).shape)
# print(np.asarray(images[1]).reshape(28,28).shape)
# print(np.asarray(images[1]).reshape(28,28,1))
# cv2.imshow('abc',np.asarray(images[1]).reshape(28,28,1).astype(np.uint8))
# cv2.imwrite('abc.png',cv2.resize(np.asarray(images[1]).reshape(28,28,1).astype(np.uint8),(32,32), interpolation = cv2.INTER_CUBIC ))
# images, labels = mndata.load_testing()





class data_loader:

	def __init__(self,config,indices):
		self.config = config


		mndata = MNIST('../data_loader/samples')
		images, labels = mndata.load_training()
		self.images_array = np.asarray([cv2.resize(np.asarray(x).reshape(28,28,1).astype(np.float32),(32,32), interpolation = cv2.INTER_CUBIC )
				for x in images
			])
		self.images_array = [self.images_array[x] for x in indices]
		self.labels = [labels[x] for x in indices]

		self.cur_index = 0
		self.no_of_batches = int(len(indices)/self.config.batch_size)
		self.total_number = self.no_of_batches*self.config.batch_size

	def get_num_batches(self):
		return self.no_of_batches

	def get_batch(self):
		data_batch = dict()

		data_batch['true_images'] = np.expand_dims(np.asarray(self.images_array[self.cur_index:self.cur_index+self.config.batch_size]),1)
		data_batch['labels'] = self.labels[self.cur_index:self.cur_index+self.config.batch_size]

		self.cur_index = (self.cur_index+self.config.batch_size)%self.total_number

		return data_batch



















