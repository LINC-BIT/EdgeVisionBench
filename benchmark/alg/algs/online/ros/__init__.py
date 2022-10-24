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
    name='ROS',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class ROS(ABOnlineDAAlgorithm):
    
    def __init__(self, models, alg_models_manager: ABAlgModelsManager, hparams: Dict, device, random_seed, res_save_dir):
        super().__init__(models, alg_models_manager, hparams, device, random_seed, res_save_dir)
        
        self.source_train_models = alg_models_manager.get_deepcopied_models(models)
    
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier', 'discriminator']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'weight_center_loss': float,
            'optimizer': str,
            'optimizer_args': dict,
            'cls_weight_source': float,
            'ss_weight_source': float,
            'use_weight_net_first_part': bool,
            'ss_weight_target': float,
            'weight_class_unknown': float,
            'entropy_weight': float
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        train(self.models, self.alg_models_manager, self.hparams, exp_tracker, 
              scenario, self.device, self.alg_models_manager.get_deepcopied_models(self.source_train_models))
