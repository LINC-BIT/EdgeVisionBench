from torch.utils.data import Dataset
import os
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image


class CommonDataset(Dataset):
    def __init__(self, images_path, labels_path, x_transform, y_transform):
        self.imgs_path = images_path
        self.labels_path = labels_path
        
        # for p in os.listdir(os.path.join(image_dir)):
        #     p = os.path.join(dataset_project_dir, 'images', p)
        #     if not p.endswith('png'):
        #         continue
        #     self.imgs_path += [p]
        #     self.labels_path += [p.replace('images', 'labels_gt')]
        
        # self.x_transform = T.Compose(
        #     [
        #         T.Resize((224, 224)),
        #         T.ToTensor(),
        #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        # self.y_transform = T.Compose(
        #     [
        #         T.Resize((224, 224)),
        #         T.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
        #     ]
        # )
        self.x_transform = x_transform
        self.y_transform = y_transform
        
        
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        x_path = os.path.join(self.imgs_path[idx])
        y_path = os.path.join(self.labels_path[idx])
        
        x = default_loader(x_path)
        # y = default_loader(y_path)
        y = Image.open(y_path).convert('L')

        x = self.x_transform(x)
        y = self.y_transform(y)
        
        return x, y
