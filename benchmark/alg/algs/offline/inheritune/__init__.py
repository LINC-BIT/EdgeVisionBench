from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm
import numpy as np
import torch.nn.functional as F
from torch import nn


@algorithm_register(
    name='Inheritune',
    stage='offline',
    supported_tasks_type=['Image Classification']
)
class Inheritune(ABOfflineTrainAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in [
            'pretrained feature extractor in feature extractor (M)', 
            'fc in feature extractor (E)',
            'classifier (G_s)',
            'aux classifier (G_n)'
        ]])
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'softmax_temperature': float,
            'classification_weight': (float, float)
        }).validate(hparams)
    
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        train_set = scenario.get_merged_source_dataset('train')
        train_loader = iter(scenario.build_dataloader(train_set, self.hparams['batch_size'], 
                                                      self.hparams['num_workers'], True, None))
        
        exp_tracker.start_train()
        
        M = self.alg_models_manager.get_model(self.models, 'pretrained feature extractor in feature extractor (M)')
        Es = self.alg_models_manager.get_model(self.models, 'fc in feature extractor (E)')
        Gs = self.alg_models_manager.get_model(self.models, 'classifier (G_s)')
        Gn = self.alg_models_manager.get_model(self.models, 'aux classifier (G_n)')
        
        optimizers = [torch.optim.__dict__[self.hparams['optimizer']](
            model.parameters(), **self.hparams['optimizer_args']) for model in [Es, Gs, Gn]]
        schedulers = [torch.optim.lr_scheduler.__dict__[
            self.hparams['scheduler']](optimizer, **self.hparams['scheduler_args']) for optimizer in optimizers]
        
        num_classes_source = scenario.get_num_classes()[0]
        
        M_spliced = self.create_negative_features(train_loader, M)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=Gn.out_features, random_state=0).fit(M_spliced.cpu().numpy())
        gt_negative_labels = num_classes_source + kmeans.labels_
        
        for iter_index in tqdm.tqdm(range(self.hparams['num_iters']), desc='iterations',
                                    leave=False, dynamic_ncols=True):
            
            x, y = next(train_loader)
            x, y = x.to(self.device), y.to(self.device)
            
            [model.train() for model in [Es, Gs, Gn]]
            
            # self.gt = Variable(torch.LongTensor(self.data['label'])).cuda().long()
            # img_source = Variable(self.data['img'][:, :3, :, :]).cuda().float()
            img_source, gt = x, y
            
            idx = np.random.randint(M_spliced.shape[0], size=self.hparams['batch_size'])
            samples_spliced_M = M_spliced[idx].cuda()
            gt_neg = torch.from_numpy(gt_negative_labels[idx]).long().cuda()
            gt = torch.cat([gt, gt_neg], dim=0)
            
            features = {}

            with torch.no_grad():
                features['M'] = M(img_source)
                features['M'] = torch.cat([features['M'], samples_spliced_M], dim=0)

            features['Es'] = Es(features['M'])
            features['Gs'] = Gs(features['Es'])
            features['Gn'] = Gn(features['Es'])
            
            concat_outputs = torch.cat([features['Gs'], features['Gn']], dim=-1)
            concat_softmax = F.softmax(concat_outputs / self.hparams['softmax_temperature'], dim=-1)
            preds = torch.argmax(concat_softmax, dim=-1)
            # src = self.settings['num_src']
            src = self.hparams['batch_size']

            # Source samples : Batchwise Accuracy
            pred_classes = preds[:src]
            gt_classes = gt[:src]
            if len(pred_classes) != 0:
                src_batch_acc = (pred_classes.float() == gt_classes.float()).float().mean()
                # self.summary_dict['acc/src_batch_acc'] = src_batch_acc
                exp_tracker.add_scalar('running/src_batch_acc', src_batch_acc, iter_index)
                
            outs = torch.cat([features['Gs'], features['Gn']], dim=-1)
			# src = self.settings['num_src']
            src_classification = nn.CrossEntropyLoss(reduction='mean')(outs[:src], gt[:src].long())
            neg_classification = nn.CrossEntropyLoss(reduction='mean')(outs[src:], gt[src:].long())
            w1, w2 = self.hparams['classification_weight']
            loss = w1 * src_classification + w2 * neg_classification
            
            [optimizer.zero_grad() for optimizer in optimizers]
            loss.backward()
            [optimizer.step() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]
            
            exp_tracker.add_losses({
                'src_cls': w1 * src_classification,
                'neg_cls': w2 * neg_classification    
            }, iter_index)
            if iter_index % 10 == 0:
                exp_tracker.add_running_perf_status(iter_index)
            if iter_index % 500 == 0:
                met_better_model = exp_tracker.add_val_accs(iter_index)
                if met_better_model:
                    exp_tracker.add_models()
        exp_tracker.end_train()
        
    def create_negative_features(self, dataloader_source, M):
        spliced_M = []

        # while len(spliced_M) < 20000:

			# dataset_source_train = TemplateDataset(self.index_list_train_source, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
			# dataloader_source = DataLoader(dataset_source_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
			
        for data in tqdm.tqdm(dataloader_source, dynamic_ncols=True, leave=False, desc='creating negative features...'):

            # x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
            # labels_source = Variable(data['label']).to(self.settings['device'])
            x, labels_source = data
            x, labels_source = x.to(self.device), labels_source.to(self.device)
            
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

            M_x = M(x).detach().cpu().numpy()
            M_x_shuffle = M(x_shuffle).detach().cpu().numpy()

            for i in range(M_x.shape[0]):
                _x = M_x[i].copy()
                _x_shf = M_x_shuffle[i].copy()

                sorted_x = np.flip(np.argsort(_x))
                top_x_idx = sorted_x[:int(0.15 * len(sorted_x))]
                bottom_x_idx = sorted_x[-int(0.10 * len(sorted_x)):]

                _x[top_x_idx] = _x_shf[top_x_idx]
                _x[bottom_x_idx] = _x_shf[bottom_x_idx]
                spliced_M.append(_x)
                
            if len(spliced_M) >= 20000:
                break

        spliced_M_feats = torch.from_numpy(np.stack(spliced_M, axis=0)).cuda()
        return spliced_M_feats
        