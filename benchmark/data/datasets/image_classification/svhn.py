from benchmark.data.datasets.data_aug import cifar_like_image_test_aug, cifar_like_image_train_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from torchvision.datasets import SVHN as RawSVHN
import numpy as np
from typing import Dict, List, Optional
from torchvision import transforms
from torchvision.transforms import Compose

from ..registery import dataset_register


@dataset_register(
    name='SVHN', 
    classes=[str(i) for i in range(10)], 
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHN(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            mean, std = [0.5] * 3, [0.5] * 3
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]) if split == 'train' else \
                transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
                
            self.transform = transform
        dataset = RawSVHN(root_dir, 'train' if split != 'test' else 'test', transform=transform, download=True)
        
        if len(ignore_classes) > 0: 
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]
        
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx
                
            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]
        
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset
