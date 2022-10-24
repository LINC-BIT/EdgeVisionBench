
from typing import Dict, List, Type, Union
from .scenario import Scenario

from schema import Schema, Optional
from torchvision.transforms import Compose

from .build import _build_scenario_info

static_scenario_registery: Dict[str, dict] = {}

def scenario_register(name, scenario_def: dict):
    Schema({
        'source_datasets_name': [str],
        'target_datasets_order': [str],
        'da_mode': lambda d: d in ['da', 'partial_da', 'open_set_da', 'universal_da'],
        'num_samples_in_each_target_domain': int,
        'data_dirs': {str: str},
        Optional('transforms'): {str: Compose},
        Optional('visualize_dir_path'): str
    }).validate(scenario_def)
    
    num_classes = _build_scenario_info(scenario_def['source_datasets_name'], scenario_def['target_datasets_order'],
                                       scenario_def['da_mode'])[-1]

    static_scenario_registery[name] = scenario_def

    return num_classes
