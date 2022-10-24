from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from torchvision.transforms import Compose


class ABDataset(ABC):
    def __init__(self, root_dir, split, transform=None, ignore_classes=[], idx_map=None):
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
        
        self.dataset = None
        
        # injected by @dataset_register
        self.name = None
        self.classes = None
        self.raw_classes = None
        self.class_aliases = None
        self.shift_type = None
        self.task_type = None # ['Image Classification', 'Object Detection', ...]
        self.object_type = None # ['generic object', 'digit and letter', ...]
    
    @abstractmethod
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        raise NotImplementedError
    
    def build(self):
        if not hasattr(self, 'classes'):
            raise AttributeError('attr `classes` is injected by `@dataset_register()`. '
                                 'Your dataset class should be wrapped with @dataset_register().')
        self.dataset = self.create_dataset(self.root_dir, self.split, self.transform, 
                                           self.classes, self.ignore_classes, self.idx_map)
        self.raw_classes = self.classes
        self.classes = [i for i in self.classes if i not in self.ignore_classes]
    
    def __getitem__(self, idx):
        if self.dataset is None:
            raise AttributeError('Real dataset is build in `@dataset_register()`. '
                                 'Your dataset class should be wrapped with @dataset_register().')
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    