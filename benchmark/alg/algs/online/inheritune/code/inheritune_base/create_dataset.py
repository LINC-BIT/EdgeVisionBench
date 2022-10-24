import os
import sys
import numpy as np
import io
from tqdm import tqdm
import dataset_utils
#Server root path
from server_path import server_root_path

from natsort import natsorted

#Place where all data is stored
dataset_dir = 'data/office_31_dataset'

dataset_exp_names = ['DtoA']
datasets_sources = [['dslr']]
datasets_targets = ['amazon']

C = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
Ct_uk = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

for dataset_exp_name, datasets_source, datasets_target in tqdm(list(zip(dataset_exp_names, datasets_sources, datasets_targets))):

	print('creating data', dataset_exp_name)

	resolution = 224

	source_train_val_split = 0.9
	target_train_val_split = 1

	# Create a folder inside the dataset experiment folder server_root_path, dataset_dir, dataset_exp_name
	if not os.path.exists(os.path.join(server_root_path, dataset_dir, dataset_exp_name)):
	    os.mkdir(os.path.join(server_root_path, dataset_dir, dataset_exp_name))
	else:
	    os.system('rm -rf ' + os.path.join(server_root_path, dataset_dir, dataset_exp_name) + '/*')
	    
	num_datasets = len(datasets_source) + 1
	all_datasets = datasets_source + [datasets_target]

	#Create Source Data
	dataset_utils.save_data(server_root_path, dataset_dir, dataset_exp_name, 'source_images', datasets_source, C, [], source_train_val_split, resolution=resolution)

	#Create Target Data
	dataset_utils.save_data(server_root_path, dataset_dir, dataset_exp_name, 'target_images', [datasets_target], C, Ct_uk, target_train_val_split, resolution=resolution)
