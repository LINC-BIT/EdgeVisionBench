import cv2
from skimage import io
import numpy as np
import os
import glob
#import img_utils
from tqdm import tqdm
from skimage import transform


def crop_and_pad(img1, size=224.0, max_=0):
    # print('## File: img_utills.py | Function: crop_and_pad ##')
    h = img1.shape[0]
    w = img1.shape[1]

    # Maintain the same aspect ratio and resize the image

    if h>w:
        ratio = float(size)/float(h)
        new_h = int(size)
        new_w = int(ratio*w)
    else:
        ratio = float(size)/float(w)
        # print(ratio)
        new_h = int(ratio*h)
        new_w = int(size)

    # print(img1.shape)
    # print('newsize', new_h, new_w)
    img_resized = cv2.resize(img1, (new_w, new_h))

    pad_h = (size-new_h)
    pad_h_start = int(pad_h//2)
    pad_h_stop = int(pad_h - pad_h_start)
    pad_w = (size-new_w)
    pad_w_start = int(pad_w//2)
    pad_w_stop = int(pad_w - pad_w_start)

    img_cropped = cv2.copyMakeBorder(img_resized,pad_h_start,pad_h_stop,pad_w_start,pad_w_stop, cv2.BORDER_CONSTANT, value=int(max_))

    return img_cropped

def save_image(img_path_read, data_dir, category, cat_it, dataset_name, iteration_no, phase, resolution):
    # print('## File: dataset_utils.py | Function: save_image ##')
    image = io.imread(img_path_read)
    crop = crop_and_pad(image, size=resolution)
    img_name = os.path.join(data_dir, phase,  '_'.join(['category', category, 'category_number', str(cat_it), 'dataset', dataset_name, str(iteration_no)])) + '.png'
    image =  image.astype('uint8')
    io.imsave(img_name, crop)

def save_data(server_root_path, dataset_dir, dataset_exp_name, images_folder_name, datasets, C, C_dash, train_val_split, resolution):
    # print('## File: dataset_utils.py | Function: save_data ##')
    data_dir = os.path.join(server_root_path, dataset_dir, dataset_exp_name, images_folder_name)
    # print data_dir
    if os.path.exists(data_dir):
        os.system('rm -rf ' + data_dir + '/*')
    else:
        os.mkdir(data_dir)

    categories = list(C)
    categories.extend(C_dash)
    
    os.mkdir(os.path.join(data_dir, 'train'))
    os.mkdir(os.path.join(data_dir, 'val'))
    train_iteration_no = 0
    val_iteration_no = 0
    
    for cat_it, category in tqdm(list(enumerate(categories))):

        for dataset_name in datasets:

            imgs_path = np.array(glob.glob(os.path.join(server_root_path, dataset_dir, dataset_name, 'images', category) + '/*'))
            np.random.shuffle(imgs_path)
            split_pos = int(train_val_split*len(imgs_path))
        
            imgs_path_train = imgs_path[:split_pos]
            imgs_path_val = imgs_path[split_pos:]
            
            #Save train images
            for img_path_read in imgs_path_train:
                # print(img_path_read)
                save_image(img_path_read, data_dir, category, cat_it, dataset_name, train_iteration_no, 'train', resolution)
                train_iteration_no = train_iteration_no + 1

            #Save val images
            for img_path_read in imgs_path_val:
                val_iteration_no = val_iteration_no + 1    
                save_image(img_path_read, data_dir, category, cat_it, dataset_name, val_iteration_no, 'val', resolution)    
         
    temp_paths_train = os.listdir(os.path.join(data_dir, 'train'))
    temp_paths_train = [os.path.join(dataset_dir, dataset_exp_name, images_folder_name, 'train', x) for x in temp_paths_train]
    imgs_path_train = np.array(temp_paths_train)
    
    temp_paths_val = os.listdir(os.path.join(data_dir, 'val'))
    temp_paths_val = [os.path.join(dataset_dir, dataset_exp_name, images_folder_name, 'val', x) for x in temp_paths_val]
    imgs_path_val = np.array(temp_paths_val)
    
    if not os.path.exists(os.path.join(server_root_path, dataset_dir, dataset_exp_name, 'index_lists')):
        os.mkdir(os.path.join(server_root_path, dataset_dir, dataset_exp_name, 'index_lists'))
    
    #Save index list train
    save_path = os.path.join(server_root_path, dataset_dir, dataset_exp_name, 'index_lists')
    np.save(save_path + '/' + images_folder_name + '_index_list_' + 'train.npy',imgs_path_train)
    
    #Save index list val
    save_path = os.path.join(server_root_path, dataset_dir, dataset_exp_name, 'index_lists')
    np.save(save_path + '/' + images_folder_name + '_index_list_' + 'val.npy', imgs_path_val)
