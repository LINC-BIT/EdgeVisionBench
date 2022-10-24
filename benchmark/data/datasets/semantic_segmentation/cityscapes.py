from benchmark.data.datasets.data_aug import cityscapes_like_image_train_aug, cityscapes_like_image_test_aug, cityscapes_like_label_aug
from torchvision.datasets import Cityscapes as RawCityscapes
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose, Lambda
import os

from ..registery import dataset_register


@dataset_register(
    name='Cityscapes', 
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
class Cityscapes(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            x_transform = cityscapes_like_image_train_aug() if split == 'train' else cityscapes_like_image_test_aug()
            y_transform = cityscapes_like_label_aug()
            self.transform = x_transform
        else:
            x_transform, y_transform = transform
        
        # images_path, labels_path = [], []
        # for p in os.listdir(os.path.join(root_dir, 'images')):
        #     p = os.path.join(root_dir, 'images', p)
        #     if not p.endswith('png'):
        #         continue
        #     images_path += [p]
        #     labels_path += [p.replace('images', 'labels_gt')]
        
        ignore_label = 255
        raw_idx_map_in_y_transform = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        
        idx_map_in_y_transform = {i: i for i in range(len(classes))}
        idx_map_in_y_transform[255] = 255
        
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
                
        # merge idx map
        final_idx_map_in_y_transform = {}
        for k1, v1 in raw_idx_map_in_y_transform.items():
            final_idx_map_in_y_transform[k1] = idx_map_in_y_transform[v1]
        idx_map_in_y_transform = final_idx_map_in_y_transform
        
        def map_class(x):
            for k, v in idx_map_in_y_transform.items():
                x[x == k] = v
            return x
        
        y_transform = Compose([
            *y_transform.transforms,
            Lambda(lambda x: map_class(x))
        ])
            
        dataset = RawCityscapes(root_dir, target_type='semantic', 
                                transform=x_transform, target_transform=y_transform)
        dataset = train_val_test_split(dataset, split)
        return dataset
