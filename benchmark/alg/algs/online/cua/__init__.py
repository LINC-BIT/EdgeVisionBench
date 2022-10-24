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
from .util import mmd_rbf


@algorithm_register(
    name='CUA',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class CUA(ABOnlineDAAlgorithm):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed, res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)
        
        self.replay_train_set = None
        self.all_replay_x, self.all_replay_y = None, None
        self.met_source_datasets_name = []
        
        self.raw_ft = copy.deepcopy(self.alg_models_manager.get_model(self.models, 'feature extractor'))
        self.raw_ft.eval()
        
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        for func in [
            'get_pseudo_y_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor', 
            'get_feature(self, models, x) -> torch.Tensor'
        ]:
            assert hasattr(self.alg_models_manager, func.split('(')[0]), \
                f'you should implement `{func}` function in the alg_models_manager.'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'num_replay_samples_each_domain': int,
            'replay_loss_alpha': float
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        ft = self.alg_models_manager.get_model(self.models, 'feature extractor')
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            ft.parameters(), **self.hparams['optimizer_args'])
        
        target_train_set = scenario.get_limited_target_train_dataset()
        target_train_loader = iter(scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'], True, None))
        source_train_set = scenario.get_source_datasets('train')
        source_train_loaders = [iter(scenario.build_dataloader(d, self.hparams['batch_size'], self.hparams['num_workers'], True, None))
                                for n, d in source_train_set.items()]
        
        if self.hparams['replay_loss_alpha'] != 0 and self.all_replay_x is None:
            loaders = {n: iter(scenario.build_dataloader(d, self.hparams['num_replay_samples_each_domain'], self.hparams['num_workers'], True, None))
                                for n, d in source_train_set.items()}
            for dataset_name, loader in loaders.items():
                if dataset_name in self.met_source_datasets_name:
                    continue
                
                cur_replay_x, cur_replay_y = next(loader)
                if self.all_replay_x is None:
                    self.all_replay_x, self.all_replay_y = cur_replay_x, cur_replay_y
                else:
                    self.all_replay_x, self.all_replay_y = torch.cat(
                        [self.all_replay_x, cur_replay_x]), torch.cat([self.all_replay_y, cur_replay_y])
                self.met_source_datasets_name += [dataset_name]
                    
            self.replay_train_set = TensorDataset(self.all_replay_x.cpu(), self.all_replay_y.cpu())
        
        if self.replay_train_set is not None:
            replay_train_loader = iter(scenario.build_dataloader(self.replay_train_set, 
                                                                min(len(self.replay_train_set), self.hparams['batch_size']), 
                                                                self.hparams['num_workers'], True, None))
        
        for iter_index in range(self.hparams['num_iters']):
            source_x, source_y = next(source_train_loaders[iter_index % len(source_train_loaders)])
            target_x, = next(target_train_loader)
            source_x, source_y, target_x = source_x.to(self.device), source_y.to(self.device), target_x.to(self.device)
            
            self.alg_models_manager.get_model(self.models, 'feature extractor').train()
            self.alg_models_manager.get_model(self.models, 'classifier').eval()
            
            source_feature = self.alg_models_manager.get_feature({'feature extractor': self.raw_ft}, source_x)
            target_feature = self.alg_models_manager.get_feature(self.models, target_x)
            
            mmd_distance = mmd_rbf(source_feature, target_feature)
            
            if self.replay_train_set is not None:
                replay_x, replay_y = next(replay_train_loader)
                replay_x, replay_y = replay_x.to(self.device), replay_y.to(self.device)
                replay_loss = self.alg_models_manager.forward_to_compute_loss(self.models, replay_x, replay_y)
            else:
                replay_loss = 0.
            
            loss = mmd_distance + self.hparams['replay_loss_alpha'] * replay_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            exp_tracker.add_losses({ 'MMD': mmd_distance, 'replay': self.hparams['replay_loss_alpha'] * replay_loss }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()
            
        if self.hparams['replay_loss_alpha'] != 0:
            cur_replay_x = None
            while True:
                target_x, = next(iter(target_train_loader))
                if cur_replay_x is None:
                    cur_replay_x = target_x
                else:
                    cur_replay_x = torch.cat((cur_replay_x, target_x))
                if cur_replay_x.size(0) > self.hparams['num_replay_samples_each_domain']:
                    cur_replay_x = cur_replay_x[0: self.hparams['num_replay_samples_each_domain']]
                    break
            cur_replay_y = self.alg_models_manager.predict(self.models, cur_replay_x.to(self.device))
            cur_replay_y = self.alg_models_manager.get_pseudo_y_from_model_output(cur_replay_y)
            
            if self.replay_train_set is None:
                self.all_replay_x, self.all_replay_y = cur_replay_x, cur_replay_y
            else:
                self.all_replay_x, self.all_replay_y = torch.cat(
                    [self.all_replay_x.cpu(), cur_replay_x.cpu()]), torch.cat([self.all_replay_y.cpu(), cur_replay_y.cpu()])
                
            self.replay_train_set = TensorDataset(self.all_replay_x.cpu(), self.all_replay_y.cpu())
