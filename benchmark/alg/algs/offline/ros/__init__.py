from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm

from .train import train


@algorithm_register(
    name='ROS',
    stage='offline',
    supported_tasks_type=['Image Classification']
)
class ROS(ABOfflineTrainAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])        
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'weight_center_loss': float,
            'optimizer': str,
            'optimizer_args': dict,
            'cls_weight_source': float,
            'ss_weight_source': float
        }).validate(hparams)
        
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        exp_tracker.start_train()
        train(self.models, self.alg_models_manager, self.hparams, exp_tracker, scenario, self.device)
        exp_tracker.end_train()
        