from benchmark.data.datasets.data_aug import cityscapes_like_image_train_aug, cityscapes_like_image_test_aug, cityscapes_like_label_aug
# from torchvision.datasets import Cityscapes as RawCityscapes
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose, Lambda
import os

from .common_dataset import VideoDataset
from ..registery import dataset_register


@dataset_register(
    name='HMDB51', 
    classes=['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup', 'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave'],
    task_type='Action Recognition',
    object_type='Web Video',
    # class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class HMDB51(ABDataset): # just for demo now
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        # if transform is None:
        #     x_transform = cityscapes_like_image_train_aug() if split == 'train' else cityscapes_like_image_test_aug()
        #     y_transform = cityscapes_like_label_aug()
        #     self.transform = x_transform
        # else:
        #     x_transform, y_transform = transform
        
        dataset = VideoDataset([root_dir], mode='train')
        
        if len(ignore_classes) > 0: 
            for ignore_class in ignore_classes:
                ci = classes.index(ignore_class)
                dataset.fnames = [img for img, label in zip(dataset.fnames, dataset.label_array) if label != ci]
                dataset.label_array = [label for label in dataset.label_array if label != ci]
                
        if idx_map is not None:
            dataset.label_array = [idx_map[label] for label in dataset.label_array]
        
        dataset = train_val_test_split(dataset, split)
        return dataset
