from ....ab_algorithm import ABOnlineDAAlgorithm
from .....scenario.scenario import Scenario
from ....registery import algorithm_register
from .....exp.exp_tracker import OnlineDATracker
from .....exp.alg_model_manager import ABAlgModelsManager
from typing import Dict, List

from schema import Schema
import torch
import torch.optim

from .train import train


@algorithm_register(
    name='ETN',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class ETN(ABOnlineDAAlgorithm):
    
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier', 
                                                 'discriminator', 'auxiliary classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            "num_iters": int,
            'batch_size': int,
            'num_workers': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict,
            'adv_loss_tradeoff': float,
            'entropy_tradeoff': float,
            'adv_loss_aug_tradeoff': float,
            'ce_aug_tradeoff': float
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        train(self.models, self.hparams, self.device, self.alg_models_manager, exp_tracker, scenario)
