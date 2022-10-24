
from typing import Dict, List, Tuple, Type, Union

from .ab_algorithm import ABOfflineTrainAlgorithm, ABOnlineDAAlgorithm
# from .algs import *


static_offline_alg_registery = {}
static_online_alg_registery = {}

def algorithm_register(name: Union[str, List[str]], stage, supported_tasks_type):
    class _Register:
        def __init__(self, func):
            self.func = func
            
            if isinstance(name, str):
                names = [name]
            else:
                names = name
            for n in names:
                if stage == 'offline':
                    static_offline_alg_registery[n] = (self, supported_tasks_type)
                else:
                    static_online_alg_registery[n] = (self, supported_tasks_type)

        def __call__(self, *args, **kwargs):
            res = self.func(*args, **kwargs)
            
            res.name = name
            
            return res
    
    return _Register
