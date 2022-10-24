import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import net
from torch.utils.data import DataLoader
from tqdm import tqdm
import subprocess
import warnings
import config_adapt as config
from Trainer_adapt import TrainerG
from data_loader import TemplateDataset

warnings.simplefilter("ignore", UserWarning)

# ======================= SANITY CHECK ======================= #

assert config.settings['running_adapt'], 'ERROR!! Config not set to run adapt trainer!!'
print('######## SANITY CHECK ########')
for key in sorted(config.settings.keys()):
	print('{}: {}'.format(key, config.settings[key]))

ip = raw_input('continue? (y/n): ')
if ip.lower() == 'y' or ip.lower() == 'yes':
	pass
else:
	print('Decided not to execute!')
	exit()

# ==================== END OF SANITY CHECK ==================== #

max_val_acc = -10000
itt_delete = []

def main():

	print('\n Setting up data sources ...')

	# ====== DELETE PAST RUNS ======
	torch.cuda.set_device(config.settings['gpu'])
	exp_name = config.settings['exp_name']
	subprocess.call(["rm", "-rf", os.path.join(config.settings['weights_path'],exp_name)])
	subprocess.call(["mkdir", os.path.join(config.settings['weights_path'],exp_name)])
	subprocess.call(["rm", "-rf", os.path.join(config.settings['summaries_path'],exp_name)])
	subprocess.call(["mkdir", os.path.join(config.settings['summaries_path'],exp_name)])
	subprocess.call(["mkdir", os.path.join(config.settings['summaries_path'],exp_name)+"/logdir_train"])
	subprocess.call(["mkdir", os.path.join(config.settings['summaries_path'],exp_name)+"/logdir_val"])

	with open(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'), 'w') as history_file:
		print('saving in ' + os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'))
		history_file.write('\n===== x ===== x =====\n')
		for key in sorted(config.settings.keys()):
			history_file.write('{}: {}\n'.format(key, config.settings[key]))
	
	# ====== DEFINE DATA SOURCES ======
	index_list_path_train_target = os.path.join(config.settings['dataset_path'], 'target_images_index_list_train.npy')
	index_list_path_val_target = os.path.join(config.settings['dataset_path'], 'target_images_index_list_train.npy')
	index_list_path_aug_target = os.path.join(config.settings['dataset_path'], 'target_images_aug_dict_train.npy')
	index_lists = [None, None, index_list_path_train_target, index_list_path_val_target, None, index_list_path_aug_target]

	# ====== CREATE NETWORK ======
	print('\n Building network ...')
	network = net.modnet(config.settings['num_C'], config.settings['num_Ct_uk'], cnn=config.settings['cnn_to_use']).cuda()
	
	# Load weights
	if config.settings['load_weights']:
		dict_to_load = torch.load(config.settings['load_weights_path'])
		for component in dict_to_load:
			if component in config.settings['to_train'].keys():
				network.components[component].load_state_dict(dict_to_load[component])

	# Initialize weights from source networks if we are loading from supervised experiment
	if config.settings['load_exp_name'].split('_')[-1] != 'adapt':
		# Initialize Ft from Fs
		network.components['Et'].load_state_dict(network.components['Es'].state_dict())
		
	# ====== DEFINE OPTIMIZERS ======
	print('\n Setting up optimizers ...')
	optimizer = {}

	for key in config.settings['use_loss']:
		if config.settings['use_loss'][key]:
			to_train = []
			for comp in config.settings['optimizer'][key]:
				if config.settings['to_train'][comp]:
					to_train.append({'params': network.components[comp].parameters(), 'lr':config.settings['lr']})
			optimizer[key] = optim.Adam(params = to_train)

	# ====== CALL TRAINING AND VALIDATION PROCESS ======
	trainval(network, optimizer, exp_name, index_lists, config.settings)


def trainval(network, optimizer, exp_name, index_lists, settings):

	global least_val_loss
	global itt_delete

	train_iter = config.settings['start_iter']
	
	trainer_G = TrainerG(network, optimizer, exp_name, index_lists, settings)
	#train_acc_list = []

	while True:

		print ("\n----------- train_iter " + str(train_iter) + ' -----------\n')
		trainer_G.set_mode_train()
		acc_gen = trainer_G.train()
		train_iter += 1 
		trainer_G.log_errors('train')
		#train_acc_list.append(acc_gen.data.cpu().numpy())

		if train_iter%config.settings['val_after'] == 0:  

			print('validating') 

			trainer_G.set_mode_val()
			min_val_flag=test(trainer_G)

			print('min_val_flag', min_val_flag)

			if(min_val_flag and trainer_G.current_iteration > config.settings['save_from']):
				print("Saving - iteration", train_iter)
				dict_to_save = {component:network.components[component].cpu().state_dict() for component in network.components}
				torch.save(dict_to_save, os.path.join(os.path.join(config.settings['weights_path'],exp_name)+'/', 'best_' + str(train_iter) + '.pth'))
				network.cuda()
				itt_delete.append(train_iter)
				if(len(itt_delete)>10):
					for k in itt_delete[:-10]:
						subprocess.call(['rm', os.path.join(os.path.join(config.settings['weights_path'],exp_name)+'/', 'best_' + str(k) + '.pth')])
					itt_delete = itt_delete[-10:]

			if train_iter >= settings['max_iter']:
				break


	#print(train_acc_list[settings['max_iter']])


def test(trainer_G, target=False, iteration=None): 

	global max_val_acc

	val_record=trainer_G.val_over_val_set()

	val_acc = val_record
	max_val_acc=max(val_acc,max_val_acc)
	trainer_G.log_errors('val')

	if(max_val_acc==val_acc):
		return True
	else:
		return False

	
if __name__ == '__main__':
	main()



