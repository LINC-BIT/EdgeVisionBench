import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms, utils
from glob import glob
from server_path import server_root_path
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from skimage import io
from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans

import config_supervised as config

from data_loader import TemplateDataset

import numpy as np

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
		[index_list_path_train_source, index_list_path_val_source, _, _, _, _] = index_lists
		self.index_list_train_source = np.load(index_list_path_train_source)
		self.index_list_val_source = np.load(index_list_path_val_source)

		# Get number of classes
		self.num_C = settings['num_C']
		self.num_Ct_uk = settings['num_Ct_uk']
		self.num_Cs = settings['num_Cs']
		self.num_Ct = settings['num_Ct']

		# Initialize data loaders
		self.get_all_dataloaders()

		if os.path.isfile('spliced_m_feats.npy'):
			spliced_M_feats = np.load('spliced_m_feats.npy')
			print('Loading spliced M feats', spliced_M_feats.shape)
			self.M_spliced = torch.from_numpy(spliced_M_feats).cuda()
		else:
			self.create_negative_features()

		if os.path.isfile('spliced_m_kmeans_labels.npy'):
			print('Loading k means labels')
			self.gt_negative_labels = np.load('spliced_m_kmeans_labels.npy')
		else:
			print('Kmeans clustering of M feats')
			kmeans = KMeans(n_clusters=45, random_state=0).fit(self.M_spliced.cpu().numpy())
			self.gt_negative_labels = self.num_C + kmeans.labels_
			np.save('spliced_m_kmeans_labels', self.gt_negative_labels)


	def create_negative_features(self):

		print('Creating M spliced features')

		spliced_M = []

		while len(spliced_M) < 20000:

			dataset_source_train = TemplateDataset(self.index_list_train_source, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
			dataloader_source = DataLoader(dataset_source_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
			
			for data in tqdm(dataloader_source):
				
				x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
				labels_source = Variable(data['label']).to(self.settings['device'])

				shuffle_order = torch.randperm(x.shape[0])
				x_shuffle = x[shuffle_order]
				label_shuffle = labels_source[shuffle_order]

				different_label_idx = (labels_source != label_shuffle)

				x = x[different_label_idx]
				x_shuffle = x_shuffle[different_label_idx]
				labels_source = labels_source[different_label_idx]
				label_shuffle = label_shuffle[different_label_idx]

				if x.shape[0] == 0:
					continue

				M_x = self.network.M(x).detach().cpu().numpy()
				M_x_shuffle = self.network.M(x_shuffle).detach().cpu().numpy()

				for i in range(M_x.shape[0]):
					_x = M_x[i].copy()
					_x_shf = M_x_shuffle[i].copy()

					sorted_x = np.flip(np.argsort(_x))
					top_x_idx = sorted_x[:int(0.15 * len(sorted_x))]
					bottom_x_idx = sorted_x[-int(0.10 * len(sorted_x)):]

					_x[top_x_idx] = _x_shf[top_x_idx]
					_x[bottom_x_idx] = _x_shf[bottom_x_idx]
					spliced_M.append(_x)

		spliced_M_feats = np.stack(spliced_M, axis=0)
		
		print('saving spliced_m_feats.npy')
		np.save('spliced_m_feats', spliced_M_feats)
		self.M_spliced = torch.from_numpy(spliced_M_feats).cuda()


	def get_all_dataloaders(self):
		
		dataset_train = TemplateDataset(self.index_list_train_source, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		self.loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

		dataset_source_val = TemplateDataset(self.index_list_val_source, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
		self.loader_source_val = DataLoader(dataset_source_val, batch_size=self.batch_size, shuffle=True, num_workers=2)


	def get_loss(self, which_loss):
	
		if which_loss == 'classification':
			outs = torch.cat([self.features['Gs'], self.features['Gn']], dim=-1)
			src = self.settings['num_src']
			src_classification = nn.CrossEntropyLoss(reduction='mean')(outs[:src], self.gt[:src].long())
			neg_classification = nn.CrossEntropyLoss(reduction='mean')(outs[src:], self.gt[src:].long())
			w1, w2 = self.settings['classification_weight']
			loss = w1 * src_classification + w2 * neg_classification

		else:
			raise NotImplementedError('Not implemented loss function ' + str(which_loss))

		self.summary_dict['loss/' + str(which_loss)] = loss.data.cpu().numpy()
		return loss


	def loss(self):

		# =========================================
		# ====== Batch Wise source accuracy =======
		# =========================================
		concat_outputs = torch.cat([self.features['Gs'], self.features['Gn']], dim=-1)
		concat_softmax = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)
		preds = torch.argmax(concat_softmax, dim=-1)
		src = self.settings['num_src']

		# Source samples : Batchwise Accuracy
		pred_classes = preds[:src]
		gt_classes = self.gt[:src]
		if len(pred_classes) != 0:
			src_batch_acc = (pred_classes.float() == gt_classes.float()).float().mean()
			self.summary_dict['acc/src_batch_acc'] = src_batch_acc

		if self.phase == 'train':

			# ====== BACKPROP LOSSES ======
			l = len(self.which_optimizer)
			current_loss = self.which_optimizer[self.current_iteration%l]

			if self.settings['use_loss'][current_loss] and self.backward:
				
				print('\nApplying loss ' + str(self.which_optimizer[self.current_iteration%l]))

				if current_loss in self.settings['losses_after_enough_iters']:
					if self.current_iteration >= self.settings['enough_iters']:
						self.optimizer[self.which_optimizer[self.current_iteration%l]].zero_grad()
						loss = self.get_loss(which_loss=self.which_optimizer[self.current_iteration%l])
						self.summary_dict['loss/' + str(self.which_optimizer[self.current_iteration%l])] = loss.cpu().detach().numpy()
						loss.backward()
						self.optimizer[self.which_optimizer[self.current_iteration%l]].step()
				elif current_loss in self.settings['losses_before_enough_iters']:
					if self.current_iteration <= self.settings['enough_iters']:
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
		
		self.gt = Variable(torch.LongTensor(self.data['label'])).cuda().long()
		img_source = Variable(self.data['img'][:, :3, :, :]).cuda().float()

		idx = np.random.randint(self.M_spliced.shape[0],size= self.settings['batch_size'])
		samples_spliced_M = self.M_spliced[idx].cuda()
		self.gt_neg = torch.from_numpy(self.gt_negative_labels[idx]).long().cuda()
		self.gt = torch.cat([self.gt, self.gt_neg], dim=0)
		
		self.features = {}

		with torch.no_grad():
			self.features['M'] = self.network.M(img_source)
			self.features['M'] = torch.cat([self.features['M'], samples_spliced_M], dim=0)

		self.features['Es'] = self.network.Es(self.features['M'])
		self.features['Gs'] = self.network.Gs(self.features['Es'])
		self.features['Gn'] = self.network.Gn(self.features['Es'])


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

		self.forward()
		self.loss()

		return self.summary_dict['acc/src_batch_acc']


	def log_errors(self, phase, iteration=None):

		for x in list(sorted(self.summary_dict.keys())):

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

		with torch.no_grad():
			self.summary_dict = {}

			dataset_source_val = TemplateDataset(self.index_list_val_source, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False, rot=False)
			dataloader_source = DataLoader(dataset_source_val, batch_size=self.batch_size, shuffle=True, num_workers=2)

			# ----------------------
			# Source validation Data
			# ----------------------

			print('\nValidating on source validation data')

			num_C = self.num_C
			num_Ct_uk = self.num_Ct_uk
			num_Cs = self.num_Cs
			num_Ct = self.num_Ct

			classes = list(range(num_C))

			avg_acc = {c:0 for c in classes}
			avg_count = {c:0 for c in classes}

			idx = -1

			for data in tqdm(dataloader_source):
				idx += 1
				x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
				labels_source = Variable(data['label']).to(self.settings['device'])

				M = self.network.components['M'](x)
				Es = self.network.components['Es'](M)
				Gs = self.network.components['Gs'](Es)
				Gn = self.network.components['Gn'](Es)

				concat_outputs = torch.cat([Gs, Gn], dim=-1)
				concat_softmax = F.softmax(concat_outputs/self.settings['softmax_temperature'], dim=-1)

				max_act, pred = torch.max(concat_softmax, dim=-1)

				for c in classes:
					avg_acc[c] += (pred[labels_source==c] == labels_source[labels_source==c]).float().sum()
					avg_count[c] += pred[labels_source==c].shape[0]

			# average accuracy
			avg = 0
			num_classes = num_C
			for c in classes:
				if avg_count[c] == 0:
					avg += 0
				else:
					avg += (float(avg_acc[c]) / float(avg_count[c]))
			avg /= float(num_classes)
			self.summary_dict['acc/source_avg'] = avg

		return self.summary_dict
