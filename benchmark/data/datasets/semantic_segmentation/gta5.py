from benchmark.data.datasets.data_aug import cityscapes_like_image_train_aug, cityscapes_like_image_test_aug, cityscapes_like_label_aug
from benchmark.data.datasets.semantic_segmentation.common_dataset import CommonDataset
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose, Lambda
import os

from ..registery import dataset_register


@dataset_register(
    name='GTA5', 
    classes=[
        'road', 'sidewalk', 'building', 'wall',
        'fence', 'pole', 'light', 'sign',
        'vegetation', 'terrain', 'sky', 'people', # person
        'rider', 'car', 'truck', 'bus', 'train',
        'motocycle', 'bicycle'
    ],
    task_type='Semantic Segmentation',
    object_type='Autonomous Driving',
    # class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class GTA5(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        x_transform, y_transform = transform
        if x_transform is None:
            x_transform = cityscapes_like_image_train_aug() if split == 'train' else cityscapes_like_image_test_aug()
            self.transform = x_transform
        if y_transform is None:
            y_transform = cityscapes_like_label_aug()
        
        images_path, labels_path = [], []
        for p in os.listdir(os.path.join(root_dir, 'images')):
            p = os.path.join(root_dir, 'images', p)
            if not p.endswith('png'):
                continue
            images_path += [p]
            labels_path += [p.replace('images', 'labels_gt')]
        
        idx_map_in_y_transform = {i: i for i in range(len(classes))}
        
        # dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0: 
            for ignore_class in ignore_classes:
                # dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                # dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
                idx_map_in_y_transform[ignore_class] = 255
                
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx
                
            # for ti, t in enumerate(dataset.targets):
            #     dataset.targets[ti] = idx_map[t]
            
            for k, v in idx_map.items():
                idx_map_in_y_transform[k] = v
        
        def map_class(x):
            for k, v in idx_map_in_y_transform.items():
                x[x == k] = v
            return x
        
        y_transform = Compose([
            *y_transform.transforms,
            Lambda(lambda x: map_class(x))
        ])
            
        dataset = CommonDataset(images_path, labels_path, x_transform=x_transform, y_transform=y_transform)
        
        dataset = train_val_test_split(dataset, split)
        return dataset
