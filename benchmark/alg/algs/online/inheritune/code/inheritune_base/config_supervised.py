import os
from server_path import server_root_path
from glob import glob
from natsort import natsorted

import matplotlib
# matplotlib.use('Agg')

import torch

# Settings are passed through this dictionary
settings = {}

# Paths of weights and summaries to be saved
settings['weights_path'] = os.path.join(server_root_path, 'weights')
settings['summaries_path'] = os.path.join(server_root_path, 'summaries')

# For supervised training. Set to false, if Train_supervised.py should not be executed.
settings['running_supervised'] = True

# Maximum number of iterations
settings['start_iter'] = 1
settings['val_after'] = 30
settings['enough_iters'] = 0

settings['dataset_exp_name'] = 'office_31_dataset/DtoA'
settings['source'] = 'dslr'
settings['target'] = 'amazon'

# Label set relationships
settings['C'] = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
settings['Ct_uk'] = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

settings['num_C'] = len(settings['C'])
settings['num_Ct_uk'] = len(settings['Ct_uk'])
settings['num_Cs'] = settings['num_C']
settings['num_Ct'] = settings['num_C'] + settings['num_Ct_uk']

# Batch Size and number of samples per iteration
settings['batch_size'] = 64
settings['num_src'] = settings['batch_size']

# Model parameters
settings['cnn_to_use'] = 'resnet50'
settings['Es_dims'] = 256
settings['softmax_temperature'] = 1
settings['rot_online'] = True

# Loading weights and experiment name. Change the experiment name here, to save the weights with the 
# corresponding exp_name. The weights of this can then be loaded into another experiment, by setting
# load_exp_name.
settings['load_weights'] = False
settings['load_exp_name'] = 'None'
settings['exp_name'] = 'DtoA_vendor'

# Define optimizers for the various losses.
settings['optimizer'] = {
	'classification': ['M', 'Es', 'Gs', 'Gn'],
}

settings['use_loss'] = {
	'classification': True
}

settings['losses_after_enough_iters'] = []
settings['classification_weight'] = [1, 0.2]
settings['losses_before_enough_iters'] = []

settings['to_train'] = {
	'M': False,
	'Es': True,
	'Et': False,
	'Gs': True,
	'Gn': True,
}
settings['lr'] = 1e-4

settings['gpu'] = 1
settings['device'] = 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])

if settings['load_weights']:
	best_weights = natsorted(glob(os.path.join(settings['weights_path'], settings['load_exp_name'], '*.pth')))[-1]
	settings['load_weights_path'] = best_weights

# Note: There might be a machine & dataset dependency for the following settings.

if settings['source'] == 'webcam' or settings['source'] == 'dslr':
	settings['max_iter'] = 60
elif settings['source'] == 'amazon':
	settings['max_iter'] = 90
else:
	raise Exception("Unknown source")

settings['dataset_path'] = os.path.join(server_root_path, 'data', settings['dataset_exp_name'], 'index_lists')
