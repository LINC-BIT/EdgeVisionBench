
from typing import Dict, Type

from .ab_dataset import ABDataset


static_dataset_registery = {}

# (NIPS'20) Measuring Robustness to Natural Distribution Shifts in Image Classification
POSSIBLE_SHIFT_TYPES = ['Consistency Shifts', 'Dataset Shifts', 'Adversarially Filtered Shifts',
               'Image Corruptions', 'Geometric Transformations', 'Style Transfer', 'Adversarial Examples']

def dataset_register(name, classes, task_type, object_type, class_aliases=[], shift_type: Dict[str, str]=None):
    assert shift_type is None or len(shift_type.keys()) == 1
    if shift_type is not None:
        assert list(shift_type.values())[0] in POSSIBLE_SHIFT_TYPES
            
    class _Register:
        def __init__(self, func: Type[ABDataset]):
            self.func = func
            static_dataset_registery[name] = (self, classes, task_type, object_type, class_aliases, shift_type)

        def __call__(self, *args, **kwargs):
            res = self.func(*args, **kwargs)

            res.name = name
            res.classes = classes
            res.class_aliases = class_aliases
            res.shift_type = shift_type
            res.task_type = task_type
            res.object_type = object_type
            
            res.build()
            
            return res
    
    return _Register
