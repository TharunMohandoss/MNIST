import os
class Config:
	def __init__(self):
		self.exp_name = "try" 
		self.learning_rate_generator = 1*(10**-3)
		self.learning_rate_discriminator = 1*(10**-3)
		self.num_epochs = 500
		self.data_size = 5000
		self.batch_size = 100
		self.train_size = 5000
		self.val_size = 1000
		self.test_size = 10000
		self.visualization_frequency = 1000
		self.step_result_print_frequency = 25
		if self.train_size + self.val_size + self.test_size > 50000:
			print('error invalid dataset sizes')
			exit(0)

		# derived config elements
		self.checkpoint_dir = os.path.join(
			"../experiments/", self.exp_name, "checkpoints")
		self.summary_dir = os.path.join(
			"../experiments/", self.exp_name, "summary")
		self.visualization_dir = os.path.join(
			"../Visualizations/"+self.exp_name)









