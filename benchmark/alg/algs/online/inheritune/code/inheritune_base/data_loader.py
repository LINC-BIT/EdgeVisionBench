import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms, utils
import torch
import os
from server_path import server_root_path

import config_supervised

class TemplateDataset(Dataset):
    
    def __init__(self, view_params_set, transform=None, random_choice=True, rot=None):
        
        self.view_params_set = view_params_set
        self.transform = transform
        self.data = {} 
        self.random_choice = random_choice
        self.rot_online = rot
        
    def __len__(self):
        if self.random_choice:
            return 1000000
        else:
            return len(self.view_params_set)

    def get_random_90_degree_augmentation(self, img):
        assert (img.shape == (224, 224, 3)) or (img.shape == (32, 32, 3))

        if type(img) == np.ndarray:
            img = transforms.ToPILImage()(img)

        angle = np.random.choice([-90, 90])
        new_img = transforms.functional.rotate(img, angle)

        return new_img
    
    def __getitem__(self, idx):

        if self.random_choice:
            idx = np.random.choice(np.arange(len(self.view_params_set)), 1)[0]
        img_name = os.path.join(server_root_path, self.view_params_set[idx])
        image = io.imread(img_name)
        image_cp = image
        if self.rot_online is None:
            if config_supervised.settings['rot_online']:
                image_cp = self.get_random_90_degree_augmentation(image_cp)
        else:
            if self.rot_online:
                image_cp = self.get_random_90_degree_augmentation(image_cp)
        res = transforms.Compose([transforms.ToTensor()])
        image_cp = res(image_cp)
        if self.transform:
            image = self.transform(image)
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image = norm(image)    

        self.data['img'] = torch.tensor(image)
        self.data['label'] = int(img_name.split('_')[-4])
        self.data['raw'] = image_cp
        self.data['filename'] = img_name
    
        return self.data.copy()