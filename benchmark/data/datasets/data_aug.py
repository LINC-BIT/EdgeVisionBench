from torchvision import transforms
import torch


def one_d_image_train_aug():
    mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
    return transforms.Compose([
        # transforms.Resize(40),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3)),
        transforms.Normalize(mean, std)
    ])


def one_d_image_test_aug():
    mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x] * 3)),
        transforms.Normalize(mean, std)
    ])


def cifar_like_image_train_aug(mean=None, std=None):
    if mean is None:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def cifar_like_image_test_aug(mean=None, std=None):
    if mean is None:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def imagenet_like_image_train_aug():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def imagenet_like_image_test_aug():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def cityscapes_like_image_train_aug():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def cityscapes_like_image_test_aug():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
def cityscapes_like_label_aug():
    import numpy as np
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])
