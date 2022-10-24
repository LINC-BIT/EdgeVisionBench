from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm
import os


@algorithm_register(
    name='EATA',
    stage='offline',
    supported_tasks_type=['Image Classification']
)
class EATA(ABOfflineTrainAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()
        
        for func in [
            'get_pseudo_y_from_model_output(self, model_output: torch.Tensor) -> torch.Tensor'
        ]:
            assert hasattr(self.alg_models_manager, func.split('(')[0]), \
                f'you should implement `{func}` function in the alg_models_manager.'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_fisher_samples': int,
            'optimizer': str,
            'optimizer_args': dict
        }).validate(hparams)
    
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        model = self.alg_models_manager.get_model(self.models, 'main model')
        model = model.to(self.device)
        
        train_sets = scenario.get_source_datasets('train')
        num_iters = self.hparams['num_fisher_samples'] // self.hparams['batch_size']
        train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'], 
                                                 self.hparams['num_workers'], True, True)) for n, d in train_sets.items()}

        optimizer = torch.optim.__dict__[self.hparams['optimizer']](
                model.parameters(), **self.hparams['optimizer_args'])
        
        exp_tracker.start_train()
        
        all_fishers = {}
        
        for train_loader_name, train_loader in train_loaders.items():
            
            fishers = {}
            
            for iter_index in tqdm.tqdm(range(num_iters), desc='iterations',
                                        leave=False, dynamic_ncols=True):
                
                model.train()
                self.alg_models_manager.set_model(self.models, 'main model', model)

                x, _ = next(train_loader)
                x = x.to(self.device)
                
                output = self.alg_models_manager.predict(self.models, x)
                target = self.alg_models_manager.get_pseudo_y_from_model_output(output)
                task_loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, target)

                optimizer.zero_grad()
                task_loss.backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if iter_index > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_index == num_iters - 1:
                            fisher = fisher / num_iters
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                
            all_fishers[train_loader_name] = fishers
            torch.save(all_fishers, os.path.join(self.res_save_dir, 'fishers.pth'))
            
        exp_tracker.end_train()