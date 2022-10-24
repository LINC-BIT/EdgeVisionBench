from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker
from .....exp.alg_model_manager import ABAlgModelsManager
from typing import Dict, List

from schema import Schema
import torch
import torch.optim

from .util import obtain_label


@algorithm_register(
    name='SHOT',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class SHOT(ABOnlineDAAlgorithm):
    
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'updating_pseudo_label_interval': int,
            'pseudo_label_task_loss_alpha': float,
            'im_loss_alpha': float
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        ft = self.alg_models_manager.get_model(self.models, 'feature extractor')
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            ft.parameters(), **self.hparams['optimizer_args'])
        
        target_train_set = scenario.get_limited_target_train_dataset()
        
        for iter_index in range(self.hparams['num_iters']):
            if iter_index % self.hparams['updating_pseudo_label_interval'] == 0:
                target_train_loader = scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'],
                                                                False, False)
                target_train_set_with_pseudo_label = obtain_label(target_train_loader, 
                                                                  self.alg_models_manager.get_model(self.models, 'feature extractor'), 
                                                                  self.alg_models_manager.get_model(self.models, 'classifier'))
                target_train_loader = scenario.build_dataloader(target_train_set_with_pseudo_label, self.hparams['batch_size'], self.hparams['num_workers'],
                                                                True, True)
                target_train_loader = iter(target_train_loader)
                
            x, y = next(target_train_loader)
            x, y = x.to(self.device), y.to(self.device)
            
            self.alg_models_manager.get_model(self.models, 'feature extractor').train()
            self.alg_models_manager.get_model(self.models, 'classifier').eval()

            task_loss, im_loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, y)
            loss = self.hparams['pseudo_label_task_loss_alpha'] * task_loss + self.hparams['im_loss_alpha'] * im_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            exp_tracker.add_losses({ 
                'task': self.hparams['pseudo_label_task_loss_alpha'] * task_loss, 
                'IM': self.hparams['im_loss_alpha'] * im_loss 
            }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()
            