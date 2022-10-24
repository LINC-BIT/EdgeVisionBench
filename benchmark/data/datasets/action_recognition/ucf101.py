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
    name='UCF101', 
    classes=['apply_eye_makeup', 'apply_lipstick', 'archery', 'baby_crawling', 'balance_beam', 'band_marching', 'baseball_pitch', 'basketball', 'basketball_dunk', 'bench_press', 'biking', 'billiards', 'blow_dry_hair', 'blowing_candles', 'body_weight_squats', 'bowling', 'boxing_punching_bag', 'boxing_speed_bag', 'breast_stroke', 'brushing_teeth', 'clean_and_jerk', 'cliff_diving', 'cricket_bowling', 'cricket_shot', 'cutting_in_kitchen', 'diving', 'drumming', 'fencing', 'field_hockey_penalty', 'floor_gymnastics', 'frisbee_catch', 'front_crawl', 'golf_swing', 'haircut', 'hammer_throw', 'hammering', 'handstand_pushups', 'handstand_walking', 'head_massage', 'high_jump', 'horse_race', 'horse_riding', 'hula_hoop', 'ice_dancing', 'javelin_throw', 'juggling_balls', 'jump_rope', 'jumping_jack', 'kayaking', 'knitting', 'long_jump', 'lunges', 'military_parade', 'mixing', 'mopping_floor', 'nunchucks', 'parallel_bars', 'pizza_tossing', 'playing_cello', 'playing_daf', 'playing_dhol', 'playing_flute', 'playing_guitar', 'playing_piano', 'playing_sitar', 'playing_tabla', 'playing_violin', 'pole_vault', 'pommel_horse', 'pull_ups', 'punch', 'push_ups', 'rafting', 'rock_climbing_indoor', 'rope_climbing', 'rowing', 'salsa_spin', 'shaving_beard', 'shotput', 'skate_boarding', 'skiing', 'skijet', 'sky_diving', 'soccer_juggling', 'soccer_penalty', 'still_rings', 'sumo_wrestling', 'surfing', 'swing', 'table_tennis_shot', 'tai_chi', 'tennis_swing', 'throw_discus', 'trampoline_jumping', 'typing', 'uneven_bars', 'volleyball_spiking', 'walking_with_dog', 'wall_pushups', 'writing_on_board', 'yo_yo'],
    task_type='Action Recognition',
    object_type='Web Video',
    # class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class UCF101(ABDataset): # just for demo now
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
