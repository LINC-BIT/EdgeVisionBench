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

from .train import train


@algorithm_register(
    name='OSDABP',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class OSDABP(ABOnlineDAAlgorithm):
        
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            'num_iters': int,
            'num_workers': int,
            'batch_size': int,
            'th': float,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        train(self.models, self.alg_models_manager, self.hparams, 
              exp_tracker, scenario, self.device)
