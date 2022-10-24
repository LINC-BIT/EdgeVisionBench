from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm


@algorithm_register(
    name='NormalSourceTrain',
    stage='offline',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class NormalSourceTrain(ABOfflineTrainAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict
        }).validate(hparams)
    
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        model = self.alg_models_manager.get_model(self.models, 'main model')
        model = model.to(self.device)

        optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            model.parameters(), **self.hparams['optimizer_args'])
        scheduler = torch.optim.lr_scheduler.__dict__[
            self.hparams['scheduler']](optimizer, **self.hparams['scheduler_args'])
        
        train_sets = scenario.get_source_datasets('train')
        train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'], 
                                                 self.hparams['num_workers'], True, True)) for n, d in train_sets.items()}
        
        exp_tracker.start_train()
        
        for iter_index in tqdm.tqdm(range(self.hparams['num_iters']), desc='iterations',
                                    leave=False, dynamic_ncols=True):
            
            losses = {}
            for train_loader_name, train_loader in train_loaders.items():
                model.train()
                self.alg_models_manager.set_model(self.models, 'main model', model)

                x, y = next(train_loader)
                x, y = x.to(self.device), y.to(self.device)
                
                task_loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, y)

                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                
                losses[train_loader_name] = task_loss
                
            exp_tracker.add_losses(losses, iter_index)
            if iter_index % 10 == 0:
                exp_tracker.add_running_perf_status(iter_index)
            
            scheduler.step()
            
            if iter_index % 500 == 0:
                met_better_model = exp_tracker.add_val_accs(iter_index)
                if met_better_model:
                    exp_tracker.add_models()
        exp_tracker.end_train()