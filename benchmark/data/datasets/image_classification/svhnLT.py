import torchvision
import torchvision.transforms as transforms
import os
import pickle
import scipy.io as sio
import numpy as np
from benchmark.data.datasets.data_aug import cifar_like_image_test_aug, cifar_like_image_train_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split
from torchvision.datasets import SVHN as RawSVHN
import numpy as np
from typing import Dict, List, Optional
from torchvision import transforms
from torchvision.transforms import Compose

from ..registery import dataset_register
from ..registery import longlabel
import torch

class ImbalanceSVHN(torchvision.datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='step', imb_factor=0.005, rand_number=0, split='train',
                 transform=None, target_transform=None, download=True):
        super(ImbalanceSVHN, self).__init__(root, split, transform, target_transform, download)
        self.split=split
        np.random.seed(rand_number)
        if self.split=='train':
         img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
         self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        #img_max = len(self.data) / cls_num
        img_max = 5000
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                #print('cls_idx:',cls_idx,num)
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

        targets_np = np.array(self.labels, dtype=np.int64)

        classes = np.unique(targets_np)
        # shift label 0 to the last (as original SVHN labels)
        # since SVHN itself is long-tailed, label 10 (0 here) may not contain enough images
        classes = np.concatenate([classes[1:], classes[:1]], axis=0)

        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            print(f"Class {the_class}:\t{len(idx)}")
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)

        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets
        print(new_data.shape[0])
        print( len(new_targets))
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list




cls_num_list=[]

@dataset_register(
    name='SVHNLT',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHNLT(ABDataset):
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
        self.weighted_alpha = 1
        dataset = ImbalanceSVHN(root_dir,  split='train' if split != 'test' else 'test', transform=transform, download=True)
        self.dataset1=dataset


        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]

        if split =='train' :
          cls_num_list=dataset.get_cls_num_list()
          print('cls_num_list SVHNLT:')
          print(cls_num_list)

        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


    def get_weighted_sampler(self):
        cls_num_list = self.dataset1.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.dataset1.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset1.labels), replacement=True)
        return sampler


@dataset_register(
    name='SVHNLT_001',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHNLT_001(ABDataset):
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
        self.weighted_alpha = 1
        dataset = ImbalanceSVHN(root_dir, imb_factor=0.01,split='train' if split != 'test' else 'test', transform=transform, download=True)
        self.dataset1=dataset


        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]

        if split =='train' :
          cls_num_list=dataset.get_cls_num_list()
          print('cls_num_list SVHNLT:')
          print(cls_num_list)

        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


    def get_weighted_sampler(self):
        cls_num_list = self.dataset1.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.dataset1.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset1.labels), replacement=True)
        return sampler



@dataset_register(
    name='SVHNLT_002',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHNLT_002(ABDataset):
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
        self.weighted_alpha = 1
        dataset = ImbalanceSVHN(root_dir, imb_factor=0.02,split='train' if split != 'test' else 'test', transform=transform, download=True)
        self.dataset1=dataset


        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]

        if split =='train' :
          cls_num_list=dataset.get_cls_num_list()
          print('cls_num_list SVHNLT:')
          print(cls_num_list)

        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


    def get_weighted_sampler(self):
        cls_num_list = self.dataset1.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.dataset1.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset1.labels), replacement=True)
        return sampler

@dataset_register(
    name='SVHNLT_005',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHNLT_005(ABDataset):
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
        self.weighted_alpha = 1
        dataset = ImbalanceSVHN(root_dir, imb_factor=0.05,split='train' if split != 'test' else 'test', transform=transform, download=True)
        self.dataset1=dataset


        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]

        if split =='train' :
          cls_num_list=dataset.get_cls_num_list()
          print('cls_num_list SVHNLT:')
          print(cls_num_list)

        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


    def get_weighted_sampler(self):
        cls_num_list = self.dataset1.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.dataset1.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset1.labels), replacement=True)
        return sampler


@dataset_register(
    name='SVHNLT_0005',
    classes=[str(i) for i in range(10)],
    task_type='Image Classification',
    object_type='Digit and Letter',
    class_aliases=[],
    shift_type=None
)
class SVHNLT_0005(ABDataset):
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
        self.weighted_alpha = 1
        dataset = ImbalanceSVHN(root_dir, imb_factor=0.005,split='train' if split != 'test' else 'test', transform=transform, download=True)
        self.dataset1=dataset


        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.labels != classes.index(ignore_class)]
                dataset.labels = dataset.labels[dataset.labels != classes.index(ignore_class)]

        if split =='train' :
          cls_num_list=dataset.get_cls_num_list()
          print('cls_num_list SVHNLT:')
          print(cls_num_list)

        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.labels):
                dataset.labels[ti] = idx_map[t]

        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


    def get_weighted_sampler(self):
        cls_num_list = self.dataset1.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.dataset1.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        #print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.dataset1.labels), replacement=True)
        return sampler