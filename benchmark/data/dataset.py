import importlib
from typing import Type
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from .datasets.ab_dataset import ABDataset

from .datasets import * # import all datasets
from .datasets.registery import static_dataset_registery


def get_dataset(dataset_name, root_dir, split, transform=None, ignore_classes=[], idx_map=None) -> ABDataset:
    dataset_cls = static_dataset_registery[dataset_name][0]
    dataset = dataset_cls(root_dir, split, transform, ignore_classes, idx_map)

    return dataset


def get_num_limited_dataset(dataset: ABDataset, num_samples: int, discard_label=True):
    dataloader = iter(DataLoader(dataset, num_samples // 2, shuffle=True))
    x, y = [], []
    cur_num_samples = 0
    while True:
        batch = next(dataloader)
        cur_x, cur_y = batch[0], batch[1]
        
        x += [cur_x]
        y += [cur_y]
        cur_num_samples += cur_x.size(0)
        
        if cur_num_samples >= num_samples:
            break
        
    x, y = torch.cat(x)[0: num_samples], torch.cat(y)[0: num_samples]
    if discard_label:
        new_dataset = TensorDataset(x)
    else:
        new_dataset = TensorDataset(x, y)
    
    dataset.dataset = new_dataset
    
    return dataset
