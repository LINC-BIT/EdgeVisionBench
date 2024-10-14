from benchmark.data.datasets.data_aug import one_d_image_test_aug, one_d_image_train_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from torchvision.datasets import MNIST as RawMNIST
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose

from ..registery import dataset_register

import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from ..registery import longlabel
import torch


class ImbalanceMNISTLT(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.1, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceMNISTLT, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.train = train
        np.random.seed(rand_number)
        if self.train==True:
         img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
         self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                if num>0.3*img_max:
                    longlabel.append(1)
                else:
                    longlabel.append(0)
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            #np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        #self.data = new_data
        self.data = torch.from_numpy(new_data)
        self.targets = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



@dataset_register(
    name='MNISTLT',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class MNISTLT(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = ImbalanceMNISTLT(root=root_dir, train= split != 'test', transform=transform, download=True)
        self.dataset1=dataset
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]



        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print('+++++++++++++')
        if split != 'test':
            #print("split")
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='MNISTLT_001',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class MNISTLT_001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = ImbalanceMNISTLT(root=root_dir, imb_factor=0.01,train= split != 'test', transform=transform, download=True)
        self.dataset1=dataset
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]



        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print('+++++++++++++')
        if split != 'test':
            #print("split")
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='MNISTLT_002',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class MNISTLT_002(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = ImbalanceMNISTLT(root=root_dir, imb_factor=0.02,train= split != 'test', transform=transform, download=True)
        self.dataset1=dataset
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]



        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print('+++++++++++++')
        if split != 'test':
            #print("split")
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


@dataset_register(
    name='MNISTLT_005',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class MNISTLT_005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = ImbalanceMNISTLT(root=root_dir, imb_factor=0.05,train= split != 'test', transform=transform, download=True)
        self.dataset1=dataset
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]



        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print('+++++++++++++')
        if split != 'test':
            #print("split")
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='MNISTLT_0005',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class MNISTLT_0005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = one_d_image_train_aug() if split == 'train' else one_d_image_test_aug()
            self.transform = transform
        dataset = ImbalanceMNISTLT(root=root_dir, imb_factor=0.005,train= split != 'test', transform=transform, download=True)
        self.dataset1=dataset
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]



        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print('+++++++++++++')
        if split != 'test':
            #print("split")
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()