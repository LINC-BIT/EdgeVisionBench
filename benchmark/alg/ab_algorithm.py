from torch import nn
from typing import Dict, List 
import torch
from abc import ABC, abstractmethod

from ..exp.alg_model_manager import ABAlgModelsManager
from ..exp.exp_tracker import OfflineTrainTracker, OnlineDATracker
from ..scenario.scenario import Scenario


class ABOfflineTrainAlgorithm(ABC):
    def __init__(self, models: List[nn.Module], alg_models_manager: ABAlgModelsManager,
                 hparams: Dict, device, random_seed, res_save_dir):

        self.models = models
        self.alg_models_manager = alg_models_manager
        self.hparams = hparams
        self.device = device
        self.random_seed = random_seed
        self.res_save_dir = res_save_dir
        
        self.name = None # injected
        
        self.verify_args(models, alg_models_manager, hparams)
        self.alg_models_manager.to_device(models, device)
        
    @abstractmethod
    def verify_args(self, models, alg_models_manager, hparams: dict):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        """
        We just leave this one abstract interface instead of a bunch of fine-grained interfaces 
        (e.g. `before_iteration()`, `update(x, y)`, `after_iteration()`). This greatly reduces
        the adaptation cost of existed implementation, but you should manually invoke some APIs
        to record the necessary information in the training process:
        - specific tracking APIs:
            - `exp_tracker.start_train()`
            - `exp_tracker.add_losses(losses)`: record losses of each model in each iteration
            - `exp_tracker.add_val_accs()`: record val accs of self.models in each source dataset 
                (how to test depends on `self.alg_models_manager.get_accuracy(self.models, test_dataset)`)
            - `exp_tracker.add_models()`: save self.models
            - `exp_tracker.add_running_perf_status()`
            - `exp_tracker.end_train()`
        - any TensorBoard APIs, e.g. `exp_tracker.add_scalar()`
        
        And a training progress bar (e.g. `tqdm`) is recommended.
        """
        raise NotImplementedError
    

class ABOnlineDAAlgorithm(ABC):
    def __init__(self, models, alg_models_manager: ABAlgModelsManager,
                 hparams: Dict, device, random_seed, res_save_dir):

        self.models = models
        self.alg_models_manager = alg_models_manager
        self.hparams = hparams
        self.device = device
        self.random_seed = random_seed
        self.res_save_dir = res_save_dir
        
        self.name = None # injected
        
        self.verify_args(models, alg_models_manager, hparams)
        self.alg_models_manager.to_device(models, device)
        
    @abstractmethod
    def verify_args(self, models: List[nn.Module], hparams: dict):
        raise NotImplementedError
    
    @abstractmethod
    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        """
        We just leave this one abstract interface instead of a bunch of fine-grained interfaces 
        (e.g. `before_iteration()`, `update(x, y)`, `after_iteration()`). This greatly reduces
        the adaptation cost of existed implementation, **but you should manually invoke some tracking APIs
        to record the necessary metrics in the training process**:
        - exp_tracker.add_losses()
        - exp_tracker.add_val_accs()
        """
        raise NotImplementedError
