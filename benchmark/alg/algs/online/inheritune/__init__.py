from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker
from .....exp.alg_model_manager import ABAlgModelsManager
from typing import Dict, List

from schema import Schema
import torch
import torch.optim
from torch.utils.data import TensorDataset
import copy
import tqdm
import torch.nn.functional as F
import numpy as np


@algorithm_register(
    name='Inheritune',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class Inheritune(ABOnlineDAAlgorithm):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed, res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)
        
        self.raw_Es = copy.deepcopy(self.alg_models_manager.get_model(models, 'fc in feature extractor (E)'))
        
        
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in [
            'pretrained feature extractor in feature extractor (M)', 
            'fc in feature extractor (E)',
            'classifier (G_s)',
            'aux classifier (G_n)'
        ]])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            'num_iters': int,
            'num_workers': int,
            'batch_size': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'pseudo_label_percentage': float,
            'softmax_temperature': float,
            'lambda': (float, float)
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        target_set = scenario.get_limited_target_train_dataset()
        target_train_finite_loader = scenario.build_dataloader(target_set, self.hparams['batch_size'], 
                                                      self.hparams['num_workers'], False, False)
        target_train_loader = iter(scenario.build_dataloader(target_set, self.hparams['batch_size'], 
                                                      self.hparams['num_workers'], True, None))
        
        M = self.alg_models_manager.get_model(self.models, 'pretrained feature extractor in feature extractor (M)')
        # Es = self.alg_models_manager.get_model(self.models, 'fc in feature extractor (E)')
        Es = self.raw_Es
        Et = copy.deepcopy(Es)
        Et.apply(lambda m: (m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
        self.alg_models_manager.set_model(self.models, 'fc in feature extractor (E)', Et)
        Gs = self.alg_models_manager.get_model(self.models, 'classifier (G_s)')
        Gn = self.alg_models_manager.get_model(self.models, 'aux classifier (G_n)')
        
        optimizers = [torch.optim.__dict__[self.hparams['optimizer']](
            model.parameters(), **self.hparams['optimizer_args']) for model in [Et]]
        schedulers = [torch.optim.lr_scheduler.__dict__[
            self.hparams['scheduler']](optimizer, **self.hparams['scheduler_args']) for optimizer in optimizers]
        
        num_classes_source = scenario.get_num_classes()[0]
        
        pseudo_label_train = self.get_pseudolabel_assignments(target_train_finite_loader, num_classes_source, M, Es, Gs, Gn)
        
        for iter_index in tqdm.tqdm(range(self.hparams['num_iters']), desc='iterations',
                                    leave=False, dynamic_ncols=True):
            
            x,  = next(target_train_loader)
            x = x.to(self.device)
            
            Et.train()
            
            features = {}
            
            features['M'] = M(x)
            features['Et'] = Et(features['M'])
            features['Gs'] = Gs(features['Et'])
            features['Gn'] = Gn(features['Et'])
            
            sample_M = np.random.randint(pseudo_label_train['M'].shape[0], size=self.hparams['batch_size'])
            pseudo_label_train['M_sample'] = pseudo_label_train['M'][sample_M]
            pseudo_label_train['Et_sample'] = Et(pseudo_label_train['M_sample'])
            pseudo_label_train['Gs_sample'] = Gs(pseudo_label_train['Et_sample'])
            pseudo_label_train['Gn_sample'] = Gn(pseudo_label_train['Et_sample'])
            pseudo_label_train['pseudo_label_sample'] = pseudo_label_train['pseudo_label'][sample_M]
            
            with torch.no_grad(): # To get instance-level weights
                features['w_Es'] = Es(features['M'])
                features['w_Gs'] = Gs(features['w_Es'])
                features['w_Gn'] = Gn(features['w_Es'])
                
            # loss
            concat_outputs = torch.cat([features['Gs'], features['Gn']], dim=-1)
            concat_softmax = F.softmax(concat_outputs / self.hparams['softmax_temperature'], dim=-1)

            # pred = torch.argmax(concat_softmax, dim=-1)
            # pred[pred >= (self.num_C)] = (self.num_C)

            # target_acc = (pred.float() == self.gt.float()).float().mean()
            # self.summary_dict['acc/target_acc'] = target_acc
            
            # adaptation loss
            num_Cs = num_classes_source

            concat_outputs = torch.cat([features['Gs'], features['Gn']], dim=-1)
            y_cap = F.softmax(concat_outputs/self.hparams['softmax_temperature'], dim=-1)
            w_concat_outputs = torch.cat([features['w_Gs'], features['w_Gn']], dim=-1)
            w_y_cap = F.softmax(w_concat_outputs/self.hparams['softmax_temperature'], dim=-1)
            W1, W2 = self.get_weight(w_y_cap, num_Cs)

            # Detach W from the graph
            W1 = torch.from_numpy(W1.cpu().detach().numpy()).cuda()

            # Soft binary entropy way of pushing samples to the corresponding regions
            y_cap_s = torch.sum(y_cap[:, :num_Cs], dim=-1)
            y_cap_n = 1 - y_cap_s

            Ld_1 = W1 * (-torch.log(y_cap_s)) + W2 * (-torch.log(y_cap_n))

            # Soft categorical entropy way of pushing samples to the corresponding regions
            y_tilde_s = F.softmax(features['Gs']/self.hparams['softmax_temperature'], dim=-1)
            y_tilde_n = F.softmax(features['Gn']/self.hparams['softmax_temperature'], dim=-1)

            H_s = - torch.sum(y_tilde_s * torch.log(y_tilde_s), dim=-1)
            H_n = - torch.sum(y_tilde_n * torch.log(y_tilde_n), dim=-1)

            Ld_2 = W1 * H_s + W2 * H_n
            l1, l2 = self.hparams['lambda']
            loss_over_batch = Ld_1 * l1 + Ld_2 * l2
            adaptation_loss = torch.mean( loss_over_batch , dim=0 )

            # pseudo_label_classification
            from torch import nn
            outs_pos = torch.cat([pseudo_label_train['Gs_sample'], pseudo_label_train['Gn_sample']], dim=-1)
            loss_pos = nn.CrossEntropyLoss(reduction='mean')(outs_pos, pseudo_label_train['pseudo_label_sample'].long())
            
            loss = adaptation_loss + loss_pos
    
            [optimizer.zero_grad() for optimizer in optimizers]
            loss.backward()
            [optimizer.step() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]
            
            exp_tracker.add_losses({
                'adaptation': adaptation_loss,
                'pos': loss_pos
            }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()
            
        self.alg_models_manager.set_model(self.models, 'fc in feature extractor (E)', Et)

    def get_pseudolabel_assignments(self, dataloader_target, num_classes_source, M, Et, Gs, Gn):

        # --------------
        # Target Dataset
        # --------------
        #self.set_mode_val()

        # dataset_target_train = TemplateDataset(self.index_list_train_target, transform=transforms.Compose([transforms.ToTensor()]), random_choice=False)
        # dataloader_target = DataLoader(dataset_target_train, batch_size=self.batch_size, shuffle=True, num_workers=2)

        # num_C = num_classes_source

        with torch.no_grad():

            total_concat_softmax = []
            total_labels = []
            total_M = []

            for x, in tqdm.tqdm(dataloader_target, dynamic_ncols=True, leave=False, desc='get pseudo label'):

                # x = Variable(data['img'][:, :3, :, :]).to(self.settings['device']).float()
                # labels_target = Variable(data['label']).to(self.settings['device'])
                
                x = x.to(self.device)
                
                # total_labels.append(labels_target)
                # labels_target[labels_target >= num_C] = self.num_C
                # fnames = data['filename']

                # M = self.network.components['M'](x)
                # Et = self.network.components['Et'](M)
                # Et = E
                # Gs = self.network.components['Gs'](Et)
                # Gn = self.network.components['Gn'](Et)
                
                Mx = M(x)
                x = Et(Mx)

                concat_outputs = torch.cat([Gs(x), Gn(x)], dim=-1)
                concat_softmax = F.softmax(concat_outputs/self.hparams['softmax_temperature'], dim=-1)

                total_concat_softmax.append(concat_softmax)
                total_M.append(Mx)

            return self.assign_pseudolabels_to_target_domain(total_labels, total_concat_softmax, total_M, num_classes_source)

    def assign_pseudolabels_to_target_domain(self, total_labels, total_concat_softmax, total_M, num_source_classes):

        # num_C = self.num_C
        # num_Ct_uk = self.num_Ct_uk
        # num_Cs = self.num_Cs
        num_Cs = num_source_classes
        # num_Ct = self.num_Ct
        ps_percent = self.hparams['pseudo_label_percentage']

        # tl = torch.cat(total_labels, dim=0) ##### JUST TO CHECK TODO TODO TODO
        tcs = torch.cat(total_concat_softmax, dim=0)
        tm_feats = torch.cat(total_M, dim=0)

        tp = torch.argmax( tcs, dim=-1 )
        # tp[tp>=self.num_Cs] = self.num_Cs
        # tl[tl>=self.num_Cs] = self.num_Cs

        W, _ = torch.max(tcs[:, :num_Cs], dim=-1)
        W1 = W
        sorted_idx = torch.argsort(W1, descending=True)

        sorted_tp = tp[sorted_idx]
        # sorted_tl = tl[sorted_idx]
        sorted_tm = tm_feats[sorted_idx]
        l = len(sorted_tp)
        
        pseudo_label_train = {}

        pseudo_label_train['M'] = sorted_tm[0: int(l * ps_percent)]
        # pseudo_label_train['gt_label'] = sorted_tl[0: int(l * ps_percent)]
        pseudo_label_train['pseudo_label'] = sorted_tp[0: int(l * ps_percent)]
        return pseudo_label_train
    
    def get_weight(self, concat_softmax, num_Cs):

        # num_Cs = self.num_C
        W, _ = torch.max(concat_softmax[:, :num_Cs], dim=-1)
        W = W / W.max()
        W1 = W.clone()
        W2 = 1-W
        return W1.squeeze(), W2.squeeze()