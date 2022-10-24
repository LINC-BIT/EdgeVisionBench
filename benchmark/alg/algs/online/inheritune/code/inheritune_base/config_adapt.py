import os
from server_path import server_root_path
from glob import glob
from natsort import natsorted

import matplotlib
# matplotlib.use('Agg')

import torch

settings = {}

settings['weights_path'] = os.path.join(server_root_path, 'weights')
settings['summaries_path'] = os.path.join(server_root_path, 'summaries')
settings['start_iter'] = 1

settings['dataset_exp_name'] = 'office_31_dataset/DtoA'
settings['source'] = 'dslr'
settings['target'] = 'amazon'

settings['C'] = ['back_pack', 'calculator', 'keyboard', 'monitor', 'mouse', 'mug', 'bike', 'laptop_computer', 'headphones', 'projector']
settings['Ct_uk'] = ['pen', 'phone', 'printer', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

settings['num_C'] = len(settings['C'])
settings['num_Ct_uk'] = len(settings['Ct_uk'])
settings['num_Cs'] = settings['num_C']
settings['num_Ct'] = settings['num_C'] + settings['num_Ct_uk']

settings['val_after'] = 500
settings['batch_size'] = 64

settings['num_positive_images'] = settings['batch_size']
settings['num_negative_images'] = settings['batch_size']

settings['cnn_to_use'] = 'resnet50'
settings['Es_dims'] = 256
settings['softmax_temperature'] = 1
settings['rot_online'] = False

# For adapt
settings['running_adapt'] = True
settings['load_weights'] = True
settings['load_exp_name'] = 'DtoA_vendor'
settings['exp_name'] = 'DtoA_client'

settings['optimizer'] = {
	'adaptation': ['Et'],
	'pseudo_label_classification': ['Et'],
}

settings['lambda'] = [1, 0.1]
settings['pseudo_label_percentage'] = 0.15

settings['use_loss'] = {
	'adaptation': True,
	'pseudo_label_classification' : True
}

settings['losses_after_enough_iters'] = ['pseudo_label_classification']
settings['losses_before_enough_iters'] = []

settings['to_train'] = {
	'M': False, # -> only upto a certain conv layer. Needs to be frozen. We'll retrain the later layers.
	'Es': False,
	'Et': True,
	'Gs': False,
	'Gn': False,
}

settings['gpu'] = 1
settings['device'] = 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])

if settings['load_weights']:
	settings['load_weights_path'] = natsorted(glob(os.path.join(server_root_path, 'weights', settings['load_exp_name'], '*.pth')))[-1]

settings['save_from'] = 12000

# Note: There might be a machine & dataset dependency for the following settings.

if settings['source'] == 'webcam' and settings['target'] == 'dslr':
	settings['enough_iters'] = 50
	settings['max_iter'] = 25000
	settings['lr'] = 2e-5
elif settings['source'] == 'webcam' and settings['target'] == 'amazon':
	settings['enough_iters'] = 300
	settings['max_iter'] = 15000
	settings['lr'] = 1e-5
elif settings['source'] == 'dslr' and settings['target'] == 'webcam':
	settings['enough_iters'] = 50
	settings['max_iter'] = 15000
	settings['lr'] = 1e-5
elif settings['source'] == 'dslr' and settings['target'] == 'amazon':
	settings['enough_iters'] = 300
	settings['max_iter'] = 15000
	settings['lr'] = 1e-5
elif settings['source'] == 'amazon' and settings['target'] == 'dslr':
	settings['enough_iters'] = 50
	settings['max_iter'] = 15000
	settings['lr'] = 1e-5
elif settings['source'] == 'amazon' and settings['target'] == 'webcam':
	settings['enough_iters'] = 50
	settings['max_iter'] = 15000
	settings['lr'] = 1e-5
else:
	raise Exception("Unknown target")


settings['dataset_path'] = os.path.join(server_root_path, 'data', settings['dataset_exp_name'], 'index_lists')
