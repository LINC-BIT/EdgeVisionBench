from .datasets.ab_dataset import ABDataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import math
import torch


def visualize_classes_image_classification(dataset: ABDataset, class_to_idx_map, rename_map, 
                      fig_save_path: str, num_imgs_per_class=2, max_num_classes=20, 
                      unknown_class_idx=None):
    
    idx_to_images = {}
    idx_to_class = {}
    idx_to_original_idx = {}
    
    reach_max_num_class_limit = False
    for i, (c, idx) in enumerate(class_to_idx_map.items()):
        if unknown_class_idx is not None and idx == unknown_class_idx:
            continue
        
        idx_to_images[idx] = []
        idx_to_class[idx] = c
        idx_to_original_idx[idx] = dataset.raw_classes.index(c)
        
        if unknown_class_idx is not None and len(idx_to_images.keys()) == max_num_classes - 1:
            reach_max_num_class_limit = True
            break
        if unknown_class_idx is None and len(idx_to_images.keys()) == max_num_classes:
            reach_max_num_class_limit = True
            break
        
    if unknown_class_idx is not None:
        idx_to_images[unknown_class_idx] = []
        idx_to_class[unknown_class_idx] = ['(unknown classes)']
    
    full_flags = {k: False for k in idx_to_images.keys()}
    
    i = 0
    while True:
        x, y = dataset[i]
        i += 1
        y = int(y)
        
        if full_flags[y]:
            continue
        
        idx_to_images[y] += [x]
        if len(idx_to_images[y]) == num_imgs_per_class:
            full_flags[y] = True
            
        if all(full_flags.values()):
            break
        
    shown_num_classes = len(idx_to_images.keys())
    if reach_max_num_class_limit:
        shown_num_classes += 1
    num_cols = 3
    num_rows = math.ceil(shown_num_classes / num_cols)
    
    plt.figure(figsize=(6.4, 4.8 * num_rows // 2))

    draw_i = 1
    for class_idx, imgs in idx_to_images.items():
        class_name = idx_to_class[class_idx]
            
        grid = make_grid(imgs, normalize=True)
        plt.subplot(num_rows, num_cols, draw_i)
        draw_i += 1
        
        plt.axis('off')
        img = grid.permute(1, 2, 0).numpy()
        plt.imshow(img)
        
        if unknown_class_idx is not None and class_idx == unknown_class_idx:
            plt.title(f'(unknown classes)\n'
                      f'current index: {class_idx}')
        else:
            class_i = idx_to_original_idx[class_idx]
            if class_name in rename_map.keys():
                renamed_class = rename_map[class_name]
                plt.title(f'{class_i}-th original class\n'
                        f'"{class_name}" (→ "{renamed_class}")\n'
                        f'current index: {class_idx}')
            else:
                plt.title(f'{class_i}-th original class\n'
                        f'"{class_name}"\n'
                        f'current index: {class_idx}')
        
    if reach_max_num_class_limit:
        plt.subplot(num_rows, num_cols, draw_i)
        plt.axis('off')
        plt.imshow(torch.ones_like(grid).permute(1, 2, 0).numpy())
        plt.title(f'(Show up to {max_num_classes} classes...)')
    
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    plt.clf()


def visualize_classes_in_object_detection(dataset: ABDataset, class_to_idx_map, rename_map, 
                      fig_save_path: str, num_imgs_per_class=2, max_num_classes=20, 
                      unknown_class_idx=None):
    
    idx_to_images = {}
    idx_to_class = {}
    idx_to_original_idx = {}
    
    reach_max_num_class_limit = False
    for i, (c, idx) in enumerate(class_to_idx_map.items()):
        if unknown_class_idx is not None and idx == unknown_class_idx:
            continue
        
        idx_to_images[idx] = []
        idx_to_class[idx] = c
        idx_to_original_idx[idx] = dataset.raw_classes.index(c)
        
        if unknown_class_idx is not None and len(idx_to_images.keys()) == max_num_classes - 1:
            reach_max_num_class_limit = True
            break
        if unknown_class_idx is None and len(idx_to_images.keys()) == max_num_classes:
            reach_max_num_class_limit = True
            break
        
    if unknown_class_idx is not None:
        idx_to_images[unknown_class_idx] = []
        idx_to_class[unknown_class_idx] = ['(unknown classes)']
    
    full_flags = {k: False for k in idx_to_images.keys()}
    
    # print(idx_to_images.keys())
    
    i = 0
    while True:
        # print(dataset[i])
        x, y = dataset[i][:2]
        i += 1
        
        cur_map = {}
        
        for label_info in y:
            if sum(label_info[1:]) == 0: # pad label
                break
            
            ci = label_info[0]
            # print(ci, label_info)
            
            if ci in cur_map.keys():
                continue # do not visualize multiple objects in an image
            
            if len(idx_to_images[ci]) == num_imgs_per_class:
                full_flags[ci] = True
                break
            
            idx_to_images[ci] += [(x, label_info[1:])]
            cur_map[ci] = 1
            
        if all(full_flags.values()):
            break
        
    shown_num_classes = len(idx_to_images.keys())
    if reach_max_num_class_limit:
        shown_num_classes += 1
    num_cols = 3
    num_rows = math.ceil(shown_num_classes / num_cols)
    
    plt.figure(figsize=(6.4, 4.8 * num_rows // 2))
    
    from torchvision.transforms import ToTensor
    from PIL import Image, ImageDraw
    import numpy as np
    
    def draw_bbox(img, bbox):
        img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
        draw = ImageDraw.Draw(img)
        draw.rectangle(bbox, outline=(255, 0, 0), width=6)
        return np.array(img)

    draw_i = 1
    for class_idx, imgs in idx_to_images.items():
        imgs, bboxes = [img[0] for img in imgs], [img[1] for img in imgs]
        class_name = idx_to_class[class_idx]
        
        # draw bbox
        imgs = [draw_bbox(img, bbox) for img, bbox in zip(imgs, bboxes)]
        imgs = [ToTensor()(img) for img in imgs]
        
        grid = make_grid(imgs, normalize=True)
        plt.subplot(num_rows, num_cols, draw_i)
        draw_i += 1
        
        plt.axis('off')
        img = grid.permute(1, 2, 0).numpy()
        plt.imshow(img)
        
        if unknown_class_idx is not None and class_idx == unknown_class_idx:
            plt.title(f'(unknown classes)\n'
                      f'current index: {class_idx}')
        else:
            class_i = idx_to_original_idx[class_idx]
            if class_name in rename_map.keys():
                renamed_class = rename_map[class_name]
                plt.title(f'{class_i}-th original class\n'
                        f'"{class_name}" (→ "{renamed_class}")\n'
                        f'current index: {class_idx}')
            else:
                plt.title(f'{class_i}-th original class\n'
                        f'"{class_name}"\n'
                        f'current index: {class_idx}')
        
    if reach_max_num_class_limit:
        plt.subplot(num_rows, num_cols, draw_i)
        plt.axis('off')
        plt.imshow(torch.ones_like(grid).permute(1, 2, 0).numpy())
        plt.title(f'(Show up to {max_num_classes} classes...)')
    
    plt.tight_layout()
    plt.savefig(fig_save_path, dpi=300)
    plt.clf()
