import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms, utils
from skimage import io
from server_path import server_root_path
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from glob import glob
from torchvision.transforms import ToTensor

import config_adapt as config

from data_loader import TemplateDataset


class TrainerG():

	def __init__(self, network, optimizer, exp_name, index_lists, settings):
	
		# Set the network and optimizer
		self.network = network
		self.to_train = settings['to_train']

		# Optimizers to use
		self.optimizer = optimizer
		self.which_optimizer = list(sorted(self.optimizer.keys()))
		print('\noptimizers: ' + str(self.which_optimizer) + '\n')

		# Save the settings
		self.settings = settings

		# Initialize the val and train writers
		self.val_writer = SummaryWriter(os.path.join(server_root_path, 'summaries', exp_name, 'logdir_val'))
		self.train_writer = SummaryWriter(os.path.join(server_root_path, 'summaries', exp_name, 'logdir_train'))
		
		# Extract commonly used settings
		self.batch_size = settings['batch_size']
		self.current_iteration = settings['start_iter']

		# Get the index lists
		[_, _, index_list_path_train_target, index_list_path_val_target, _, index_list_path_aug_target] = index_lists
		self.index_list_train_target = np.load(index_list_path_train_target)
		self.index_list_val_target = np.load(index_list_path_val_target)

		# Ensure augmented images from target validation set are removed
		self.index_list_val_target = [s for s in self.index_list_val_target if s.split('/')[-1].split('_')[0] == 'category']

		# Get number of classes
		self.num_C = settings['num_C']
		self.num_Ct_uk = settings['num_Ct_uk']
		self.num_Cs = settings['num_Cs']
		self.num_Ct = settings['num_Ct']

		# Initialize data loaders
		self.get_all_dataloaders()

		self.pseudo_label_train = {}
		self.get_pseudolabel_assignments()


	def get_all_dataloaders(self):

		dataset_train = TemplateDataset(self.index_list_train_target, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		self.loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

		dataset_target_val = TemplateDataset(self.index_list_val_target, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		self.loader_target_val = DataLoader(dataset_target_val, batch_size=self.batch_size, shuffle=True, num_workers=2)


	def assign_pseudolabels_to_target_domain(self, total_labels, total_concat_softmax, total_M):

		num_C = self.num_C
		num_Ct_uk = self.num_Ct_uk
		num_Cs = self.num_Cs
		num_Ct = self.num_Ct
		ps_percent = self.settings['pseudo_label_percentage']

		tl = torch.cat(total_labels, dim=0) ##### JUST TO CHECK TODO TODO TODO
		tcs = torch.cat(total_concat_softmax, dim=0)
		tm_feats = torch.cat(total_M, dim=0)

		tp = torch.argmax( tcs, dim=-1 )
		tp[tp>=self.num_Cs] = self.num_Cs
		tl[tl>=self.num_Cs] = self.num_Cs

		W, _ = torch.max(tcs[:, :num_Cs], dim=-1)
		W1 = W
		sorted_idx = torch.argsort(W1, descending=True)

		sorted_tp = tp[sorted_idx]
		sorted_tl = tl[sorted_idx]
		sorted_tm = tm_feats[sorted_idx]
		l = len(sorted_tl)

		self.pseudo_label_train['M'] = sorted_tm[0: int(l * ps_percent)]
		self.pseudo_label_train['gt_label'] = sorted_tl[0: int(l * ps_percent)]
		self.pseudo_label_train['pseudo_label'] = sorted_tp[0: int(l * ps_percent)]

	
	def get_pseudolabel_assignments(self):

		# --------------
		# Target Dataset
		# --------------
		#self.set_mode_val()
		
		dataset_target_train = TemplateDataset(self.index_list_train_target, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		dataloader_target = DataLoader(dataset_target_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

		num_C = self.num_C
		num_Ct_uk = self.num_Ct_uk
		num_Cs = self.num_Cs
		num_Ct = self.num_Ct

		with torch.no_grad():

			total_concat_softmax = []
			total_labels = []
			total_M = []

			for data in tqdm(dataloader_target):
				
				x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
				labels_target = Variable(data['label']).to(self.settings['device'])
				total_labels.append(labels_target)
				labels_target[labels_target>=num_C] = self.num_C
				# fnames = data['filename']

				M = self.network.components['M'](x)
				Et = self.network.components['Et'](M)
				Gs = self.network.components['Gs'](Et)
				Gn = self.network.components['Gn'](Et)

				concat_outputs = torch.cat([Gs, Gn], dim=-1)
				concat_softmax = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)

				total_concat_softmax.append(concat_softmax)
				total_M.append(M)

			self.assign_pseudolabels_to_target_domain(total_labels, total_concat_softmax, total_M)


	def get_weight(self, concat_softmax):

		num_Cs = self.num_C
		W, _ = torch.max(concat_softmax[:, :num_Cs], dim=-1)
		W = W / W.max()
		W1 = W.clone()
		W2 = 1-W
		return W1.squeeze(), W2.squeeze()


	def get_loss(self, which_loss):

		if which_loss == 'adaptation':

			num_Cs = self.num_C

			concat_outputs = torch.cat([self.features['Gs'], self.features['Gn']], dim=-1)
			y_cap = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)
			
			w_concat_outputs = torch.cat([self.features['w_Gs'], self.features['w_Gn']], dim=-1)
			w_y_cap = F.softmax(w_concat_outputs/self.settings['softmax_temperature'], dim=-1)
			W1, W2 = self.get_weight(w_y_cap)

			# Detach W from the graph
			W1 = torch.from_numpy(W1.cpu().detach().numpy()).cuda()

			# Soft binary entropy way of pushing samples to the corresponding regions
			y_cap_s = torch.sum(y_cap[:, :num_Cs], dim=-1)
			y_cap_n = 1 - y_cap_s

			Ld_1 = W1 * (-torch.log(y_cap_s)) + W2 * (-torch.log(y_cap_n))

			# Soft categorical entropy way of pushing samples to the corresponding regions
			y_tilde_s = F.softmax(self.features['Gs']/self.settings['softmax_temperature'], dim=-1)
			y_tilde_n = F.softmax(self.features['Gn']/self.settings['softmax_temperature'], dim=-1)

			H_s = - torch.sum(y_tilde_s * torch.log(y_tilde_s), dim=-1)
			H_n = - torch.sum(y_tilde_n * torch.log(y_tilde_n), dim=-1)

			Ld_2 = W1 * H_s + W2 * H_n

			l1, l2 = self.settings['lambda']

			loss_over_batch = Ld_1 * l1 + Ld_2 * l2

			loss = torch.mean( loss_over_batch , dim=0 )

		elif which_loss == 'pseudo_label_classification':
			outs_pos = torch.cat([self.pseudo_label_train['Gs_sample'], self.pseudo_label_train['Gn_sample']], dim=-1)
			loss_pos = nn.CrossEntropyLoss(reduction='mean')(outs_pos, self.pseudo_label_train['pseudo_label_sample'].long())
			loss = loss_pos

		else:
			raise NotImplementedError('Not implemented loss function ' + str(which_loss))

		self.summary_dict['loss/' + str(which_loss)] = loss.data.cpu().numpy()
		return loss


	def loss(self):

		# ==================================
		# ====== Accuracy over images ======
		# ==================================
		
		# Target Accuracy - all images
		concat_outputs = torch.cat([self.features['Gs'], self.features['Gn']], dim=-1)
		concat_softmax = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)

		pred = torch.argmax(concat_softmax, dim=-1)
		pred[pred >= (self.num_C)] = (self.num_C)

		target_acc = (pred.float() == self.gt.float()).float().mean()
		self.summary_dict['acc/target_acc'] = target_acc

		if self.phase == 'train':

			# ====== BACKPROP LOSSES ======
			l = len(self.which_optimizer)
			current_loss = self.which_optimizer[self.current_iteration%l]
			if self.settings['use_loss'][current_loss] and self.backward:
				print('\nApplying loss ' + str(self.which_optimizer[self.current_iteration%l]))

				if current_loss in self.settings['losses_after_enough_iters']:
					if self.current_iteration >= self.settings['enough_iters']:
						# print('{} >= {}'.format(self.current_iteration, self.settings['val_after']))
						self.optimizer[self.which_optimizer[self.current_iteration%l]].zero_grad()
						loss = self.get_loss(which_loss=self.which_optimizer[self.current_iteration%l])
						self.summary_dict['loss/' + str(self.which_optimizer[self.current_iteration%l])] = loss.cpu().detach().numpy()
						loss.backward()
						self.optimizer[self.which_optimizer[self.current_iteration%l]].step()
				else:
					self.optimizer[self.which_optimizer[self.current_iteration%l]].zero_grad()
					loss = self.get_loss(which_loss=self.which_optimizer[self.current_iteration%l])
					loss.backward()
					self.optimizer[self.which_optimizer[self.current_iteration%l]].step()

		self.current_iteration += 1


	def forward(self):

		# Used for evaluation purposes
		self.gt = Variable(torch.LongTensor(self.data['label'])).cuda().float()
		self.img_target = Variable(self.data['img'][:, :3, :, :]).cuda().float()
		self.gt[self.gt >= self.num_C] = (self.num_C) # Club all the target private classes into an unknown class

		self.features = {}

		# Unlabeled Target data
		self.features['M'] = self.network.M(self.img_target)
		self.features['Et'] = self.network.Et(self.features['M'])
		self.features['Gs'] = self.network.Gs(self.features['Et'])
		self.features['Gn'] = self.network.Gn(self.features['Et'])

		# Pseudolabeled target samples
		sample_M = np.random.randint(self.pseudo_label_train['M'].shape[0], size=self.batch_size)
		self.pseudo_label_train['M_sample'] = self.pseudo_label_train['M'][sample_M]
		self.pseudo_label_train['Et_sample'] = self.network.Et(self.pseudo_label_train['M_sample'])
		self.pseudo_label_train['Gs_sample'] = self.network.Gs(self.pseudo_label_train['Et_sample'])
		self.pseudo_label_train['Gn_sample'] = self.network.Gn(self.pseudo_label_train['Et_sample'])
		self.pseudo_label_train['pseudo_label_sample'] = self.pseudo_label_train['pseudo_label'][sample_M]

		with torch.no_grad(): # To get instance-level weights
			self.features['w_Es'] = self.network.Es(self.features['M'])
			self.features['w_Gs'] = self.network.Gs(self.features['w_Es'])
			self.features['w_Gn'] = self.network.Gn(self.features['w_Es'])


	def train(self):

		self.phase = 'train'

		self.summary_dict = {}

		try:
			self.data = self.dataloader_train.next()[1]
			if self.data['img'].shape[0] < self.settings['batch_size']:
				self.dataloader_train = enumerate(self.loader_train)
				self.data = self.dataloader_train.next()[1]
		except:
			self.dataloader_train = enumerate(self.loader_train)
			self.data = self.dataloader_train.next()[1]

		l = len(self.which_optimizer)
		current_loss = self.which_optimizer[self.current_iteration%l]

		#if (current_loss in self.settings['losses_after_enough_iters']) and (self.current_iteration < self.settings['enough_iters']):
		#	self.set_mode_val()
		#elif (current_loss in self.settings['losses_before_enough_iters']) and (self.current_iteration > self.settings['enough_iters']):
		#	self.set_mode_val()
		#else:
		#	self.set_mode_train()

		self.forward()
		self.loss()

		return self.summary_dict['acc/target_acc']


	def log_errors(self, phase, iteration=None):
		
		for x in self.summary_dict.keys():
			if phase == 'val':
				self.val_writer.add_scalar(x, self.summary_dict[x], self.current_iteration)
			elif phase == 'train':
				self.train_writer.add_scalar(x, self.summary_dict[x], self.current_iteration)    


	def set_mode_val(self):

		self.network.eval()
		self.backward = False
		for p in self.network.parameters():
			p.requires_grad = False
			p.volatile = True


	def set_mode_train(self):

		self.network.train()
		self.backward = True
		for p in self.network.parameters():
			p.requires_grad = True
			p.volatile = False  
		
		for comp in self.settings['to_train']:
			if self.settings['to_train'][comp] == False:
				self.network.components[comp].eval()
				for p in self.network.components[comp].parameters():
					p.requires_grad = False
					p.volatile = True	


	def val_over_val_set(self):

		self.summary_dict = {}

		# --------------
		# Target Dataset
		# --------------

		print('\nValidating on target validation data')

		dataset_target_val = TemplateDataset(self.index_list_val_target, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		dataloader_target = DataLoader(dataset_target_val, batch_size=self.batch_size, shuffle=True, num_workers=2)

		num_C = self.num_C
		num_Ct_uk = self.num_Ct_uk
		num_Cs = self.num_Cs
		num_Ct = self.num_Ct

		with torch.no_grad():

			# Running calculations of accuracy

			private_acc = 0
			private_count = 0

			classes = list(range(num_C))
			classes.append(num_C) # Unknown class

			total_count = {c:0 for c in classes}
			total_correct = {c:0 for c in classes}

			idx = -1

			total_concat_softmax = []
			total_labels = []
			total_M = []

			for data in tqdm(dataloader_target):
				idx += 1
				x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
				labels_target = Variable(data['label']).to(self.settings['device'])
				total_labels.append(labels_target)
				labels_target[labels_target>=num_C] = self.num_C # The index corresponding to the Gn logit

				M = self.network.components['M'](x)
				Et = self.network.components['Et'](M)
				Gs = self.network.components['Gs'](Et)
				Gn = self.network.components['Gn'](Et)

				concat_outputs = torch.cat([Gs, Gn], dim=-1)
				concat_softmax = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)
				total_concat_softmax.append(concat_softmax)
				total_M.append(M)

				max_act, pred = torch.max(concat_softmax, dim=-1)
				
				pred[pred>=(num_Cs)] = self.num_C # Club all the negative classes into one

				private_count += pred[labels_target>=num_C].shape[0]

				for c in classes: # for all the shared and the unknown labels
					total_correct[c] += (pred[labels_target==c] == labels_target[labels_target==c]).float().sum()
					total_count[c] += pred[labels_target==c].shape[0]

				private_acc += (pred[labels_target>=num_C] == labels_target[labels_target>=num_C]).float().sum()

			self.summary_dict['acc/priv'] = float(private_acc) / float(private_count)

			# average accuracy
			a_avg = 0
			num_classes = num_C + 1
			classes = list(range(num_C))
			classes.append(num_C)
			for c in classes:
				if total_count[c] == 0:
					a_avg += 0
				else:
					a_avg += (float(total_correct[c]) / float(total_count[c]))
			a_avg /= float(num_classes)
			self.summary_dict['acc/os'] = a_avg

		return self.summary_dict['acc/os']
