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
import torch


class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='step', imb_factor=0.005, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        print(root)
        super(ImbalanceCIFAR10, self).__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)
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

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class SemiSupervisedImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    unlabel_size_factor = 5

    def __init__(self, root, imb_type='exp', imb_factor=0.01, unlabel_imb_factor=1,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(SemiSupervisedImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, False)
        # unlabeled
        self.unlabeled_data = os.path.join(root, 'ti_80M_selected.pickle')  # selected data from 80M-TI
        self.unlabeled_pseudo = os.path.join(root,
                                             'pseudo_labeled_cifar.pickle')  # pseudo-label using model trained on imbalanced data
        self.imb_factor = imb_factor
        self.unlabel_imb_factor = unlabel_imb_factor
        self.num_per_cls_dict = dict()

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(img_num_list, img_num_list_unlabeled)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls, img_num_per_cls_unlabeled):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        print(f"Labeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

        # unlabeled dataset
        print("Loading unlabeled data from %s" % self.unlabeled_data)
        with open(self.unlabeled_data, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['data']
        aux_truth = aux['extrapolated_targets']
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_targets = pickle.load(f)
        aux_targets = aux_targets['extrapolated_targets']

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            # ground truth is only used to select samples
            idx = np.where(aux_truth == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(aux_data[selec_idx, ...])
            # append pseudo-label
            new_targets.extend(aux_targets[selec_idx])
            for pseudo_class in aux_targets[selec_idx]:
                self.num_per_cls_dict[pseudo_class] += 1
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'
        print(f"Unlabeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



@dataset_register(
    name='CIFAR10LT',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10LT(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset = ImbalanceCIFAR10(root_dir, train=True if split == 'train' else False, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]

        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)

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


    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler


@dataset_register(
    name='CIFAR10LT_001',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10LT_001(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset = ImbalanceCIFAR10(root_dir, imb_factor=0.01,train=True if split == 'train' else False, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]

        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)

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


    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

@dataset_register(
    name='CIFAR10LT_002',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10LT_002(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset = ImbalanceCIFAR10(root_dir, imb_factor=0.02,train=True if split == 'train' else False, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]

        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)

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


    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

@dataset_register(
    name='CIFAR10LT_005',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10LT_005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset = ImbalanceCIFAR10(root_dir, imb_factor=0.05,train=True if split == 'train' else False, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]

        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)

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


    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler

@dataset_register(
    name='CIFAR10LT_0005',
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    task_type='Image Classification',
    object_type='Generic Object',
    #class_aliases=[['automobile', 'car']],
    class_aliases=[],
    shift_type=None
)
class CIFAR10LT_0005(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
            self.transform = transform
        dataset = ImbalanceCIFAR10(root_dir, imb_factor=0.005,train=True if split == 'train' else False, transform=transform)
        self.dataset1 = dataset

        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0:
            for ignore_class in ignore_classes:
                #print(classes.index(ignore_class),dataset.targets,dataset.targets != classes.index(ignore_class),len(dataset.targets),len(dataset.data))
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]

        if split=='train':
            cls_num_list=dataset.get_cls_num_list()
            print('cls_num_list cifar10LT:')
            print(cls_num_list)

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


    def get_weighted_sampler(self):
        cls_num_list = self.get_cls_num_list()
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler