from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker
from .....exp.alg_model_manager import ABAlgModelsManager
from typing import Dict, List

from schema import Schema
import torch
import torch.optim

from .util import check_model, configure_model, collect_params


@algorithm_register(
    name='Tent',
    stage='online',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class Tent(ABOnlineDAAlgorithm):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed, res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)

        model = alg_models_manager.get_model(models, 'main model')
        configure_model(model)
        check_model(model)
        params, _ = collect_params(model)

        # use the same optimizer in all target domains so init it in the beginning
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            params, **self.hparams['optimizer_args'])
        
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()
        assert len(models['main model']) > 0, 'pass the path of model file in'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        target_train_set = scenario.get_limited_target_train_dataset()
        target_train_loader = scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'],
                                                        True, True)
        target_train_loader = iter(target_train_loader)

        for iter_index in range(self.hparams['num_iters']):
            x, = next(target_train_loader)
            x = x.to(self.device)

            configure_model(self.alg_models_manager.get_model(self.models, 'main model'))
            loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, None)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            exp_tracker.add_losses({ 'entropy': loss }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()
            