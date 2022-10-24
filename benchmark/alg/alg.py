from torch import nn
from typing import List
import torch

from .registery import static_offline_alg_registery, static_online_alg_registery
from .algs import *


def get_algorithm(alg_name, stage, models: List[nn.Module], alg_models_manager,
                  hparams, device, random_seed, res_save_dir):
    
    registery = static_online_alg_registery if stage == 'online' else static_offline_alg_registery
    
    alg_cls, supported_tasks_type = registery[alg_name]
    
    models = {n: torch.load(p) if isinstance(p, str) else p for n, p in models.items()}
    alg = alg_cls(models, alg_models_manager, hparams, device, random_seed, res_save_dir)
    
    return alg, supported_tasks_type
