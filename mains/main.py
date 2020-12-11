from configs.config import Config
from data_loader.data_loader import data_loader
from utils.config import process_config
from utils.dirs import create_dirs
from trainers.trainer import Trainer
from testers.tester import Tester
import os
from models.model import Model

def main():		
	print('begin preprocessing')

	config = Config()

	train_indices = range(config.train_size)
	# val_indices = [x+config.train_size for x in range(config.val_size)]
	# test_indices = [x+config.train_size+config.val_size for x in range(config.test_size)]

	train_loader = data_loader(config,train_indices)
	# val_loader = data_loader(config,val_indices)
	# test_loader = data_loader(config,test_indices)

	# create the experiments dirs
	create_dirs([config.summary_dir, config.checkpoint_dir, config.visualization_dir])

	model = Model(config)

	print('preprocessing done')

	#Trainer start
	trainer = Trainer( model, train_loader, config, None)
	trainer.train()

if __name__ == "__main__":
	main()







