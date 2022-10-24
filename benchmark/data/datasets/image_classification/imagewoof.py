from benchmark.data.datasets.data_aug import imagenet_like_image_train_aug, imagenet_like_image_test_aug
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split, train_val_test_split
from torchvision.datasets import ImageFolder
import os
from typing import Dict, List, Optional
from torchvision.transforms import Compose

from ..registery import dataset_register

@dataset_register(
    name='Imagewoof',
    classes=["Australian terrier", "Border terrier", "Samoyed", "Beagle", "Shih-Tzu", "English foxhound", "Rhodesian ridgeback", "Dingo", "Golden retriever", "Old English sheepdog"],
    task_type='Image Classification',
    object_type='Dog',
    class_aliases=[],
    shift_type=None
)

class Imagewoof(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose],
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = imagenet_like_image_train_aug() if split == 'train' else imagenet_like_image_test_aug()
            self.transform = transform
        #root_dir = os.path.join(root_dir, 'train' if split != 'test' else 'val')
        dataset = ImageFolder(root_dir, transform=transform)

        if len(ignore_classes) > 0:
            ignore_classes_idx = [classes.index(c) for c in ignore_classes]
            dataset.samples = [s for s in dataset.samples if s[1] not in ignore_classes_idx]

        if idx_map is not None:
            dataset.samples = [(s[0], idx_map[s[1]]) if s[1] in idx_map.keys() else s for s in dataset.samples]

        dataset = train_val_test_split(dataset, split)
        return dataset
