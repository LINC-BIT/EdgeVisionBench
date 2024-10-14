from benchmark.data.datasets.data_aug import cifar_like_image_train_aug, cifar_like_image_test_aug
from benchmark.data.datasets.ab_dataset import ABDataset
from benchmark.data.datasets.dataset_split import train_val_split
from torchvision.datasets import CIFAR10 as RawCIFAR10
import numpy as np
from typing import Dict, List, Optional
from torchvision.transforms import Compose
import torchvision.transforms as CCompose
from benchmark.data.datasets.registery import dataset_register
from torch.utils.data import Dataset
import random
from PIL import Image
import os

import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np
from ..registery import longlabel

import os
import PIL
import torch
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    # 可以添加其他预处理步骤，根据需要
])

# 定义 CIFAR-10-C 数据集路径
cifar10c_path = "./datasets/CIFAR10C"

# 创建 CIFAR-10-C 数据集
import numpy as np
import os
import random
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

"""
Please download data on
https://zenodo.org/record/2535967
"""


corruptions = [
    'glass_blur',
    'gaussian_noise',
    'shot_noise',
    'speckle_noise',
    'impulse_noise',
    'defocus_blur',
    'gaussian_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
    'spatter',
    'saturate',
    'frost',
]


class CIFAR10C(datasets.VisionDataset):
    def __init__(self,
                 name: str,
                 root: str = './datasets/CIFAR10C/CIFAR-10-C/',
                 transform=None, target_transform=None):
        assert name in corruptions
        super().__init__(root=root, transform=transform,target_transform=target_transform)
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        # if you want to only test a small mount of data, uncomment the following codes
        # self.data = np.concatenate([self.data[0:1000], self.data[10000:11000],
        #                             self.data[20000:21000], self.data[30000:31000],
        #                             self.data[40000:41000]])
        # self.targets = np.concatenate([self.targets[0:1000], self.targets[10000:11000],
        #                                self.targets[20000:21000], self.targets[30000:31000],
        #                                self.targets[40000:41000]])

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)




class ImbalanceCIFAR10C(CIFAR10C):
    cls_num = 10

    def __init__(self, name, imb_type='step', imb_factor=0.005, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super(ImbalanceCIFAR10C, self).__init__(name=name,  transform=transform, target_transform=target_transform)
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
        self.data = new_data
        self.targets = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list






@dataset_register(
    name='CIFAR10C',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10C(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset =  ImbalanceCIFAR10C(
                   name='snow', transform=transform
                )
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        '''
        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)
        '''
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print(split,'+++++')
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()


@dataset_register(
    name='CIFAR10C_001',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10C_001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset =  ImbalanceCIFAR10C(
                   name='snow', imb_factor=0.01 ,transform=transform
                )
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        '''
        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)
        '''
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print(split,'+++++')
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='CIFAR10C_002',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10C_002(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset =  ImbalanceCIFAR10C(
                   name='snow', imb_factor=0.02 ,transform=transform
                )
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        '''
        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)
        '''
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print(split,'+++++')
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='CIFAR10C_005',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10C_005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset =  ImbalanceCIFAR10C(
                   name='snow', imb_factor=0.05 ,transform=transform
                )
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        '''
        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)
        '''
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print(split,'+++++')
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()

@dataset_register(
    name='CIFAR10C_0005',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10C_0005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset =  ImbalanceCIFAR10C(
                   name='snow', imb_factor=0.005 ,transform=transform
                )
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        '''
        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)
        '''
        if idx_map is not None:
            # note: the code below seems correct but has bug!
            # for old_idx, new_idx in idx_map.items():
            #     dataset.targets[dataset.targets == old_idx] = new_idx

            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        #print(split,'+++++')
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset


    def get_clsnum(self):
        return self.dataset1.get_cls_num_list()