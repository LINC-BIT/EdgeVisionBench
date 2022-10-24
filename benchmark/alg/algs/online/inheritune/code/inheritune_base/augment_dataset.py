import cv2
import glob
from skimage import io
import numpy as np
import os
from tqdm import tqdm

import torch
import torchvision

chop_distances = {}

def get_chop_distance(rotated_mat, angle):

    '''
        Returns the distance at which the rotated image should be chopped off. Used in rotateImage() function.
    '''

    global chop_distances

    if angle in chop_distances.keys():
        return chop_distances[angle]

    if angle > 0:

        x = 0
        y = 0

        while(rotated_mat[y, x, 0] == 0):
            y += 1

        chop_distances[angle] = y

        # print(chop_distances)
        return y

    else:

        x = rotated_mat.shape[1] - 1
        y = 0

        while(rotated_mat[y, x, 0] == 0):
            y += 1

        chop_distances[angle] = y

        # print(chop_distances)
        return y


def rotateImage(mat, angle):
    
    '''
        Rotates an image (angle in degrees) and zooms into the image to avoid crop borders.
    '''

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to original) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    H, W, c = rotated_mat.shape

    d = get_chop_distance(rotated_mat, angle)
    chopped_image = rotated_mat[d : H-d, d : W-d]

    resized_image = cv2.resize(chopped_image, (224, 224))

    return resized_image


def random_crop(image, cropshape, padsize):
    
    '''
        Takes a random crop from the image.
    '''

    # crop height, width, channels
    H, W, C = image.shape
    p = padsize

    # Check shapes etc.
    if type(cropshape) == int:
        cH = cW = cropshape
        assert cH <= H, 'Crop size is greater than image size'
    elif len(cropshape) == 2:
        cH, cW = cropshape
        assert cH <= H and cW <= W, 'Crop size is greater than image size'
    else:
        raise Exception('Wrong crop shape (use either int (s) or tuple (h, w))')

    if type(padsize) == int:
        pH = pW = padsize
    elif len(padsize) == 2:
        pH, pW = padsize
    else:
        raise Exception('Wrong pad shape (use either int (s) or tuple (h, w))')

    # Created padded image
    paddedimage = np.zeros((cH + 2*pH, cW + 2*pW, C), dtype = image.dtype)
    paddedimage[pH:pH+H, pW:pW+W, :] = image

    # Output image
    outimage = np.zeros((cH, cW, C), dtype = image.dtype)

    # Randomly chose a start location (this is the random step)
    startx = np.random.randint(2*padsize)
    starty = np.random.randint(2*padsize)

    # Crop (H, W, C) shaped image from start locations
    outimage = paddedimage[starty:starty+cH, startx:startx+cH, :]

    return outimage


def random_horizontal_flip(image, always_flip=True):

    '''
        Randomly flips the given image.
    '''

    if always_flip:
        r = 1 # always flip
    else:
        r = np.random.rand()

    if r >= 0:
        return np.fliplr(image), True
    else:
        return np.array(image), False


def rgb_flip(img, reorder):

    '''
        Flips the channels in the given order.
    '''

    newim = img.copy()
    newim[:, :, 0] = img[:, :, reorder[0]]
    newim[:, :, 1] = img[:, :, reorder[1]]
    newim[:, :, 2] = img[:, :, reorder[2]]

    return newim


def color_jitter(img, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25):

    '''
        Applies color jitter augmentation (brightness, contrast, saturation, hue).
    '''

    tensorImage = torchvision.transforms.ToTensor()(img)
    pilImage = torchvision.transforms.ToPILImage(mode='RGB')(tensorImage)
    jitteredImage = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)(pilImage)
    jitteredArray = np.array(jitteredImage)

    return jitteredArray


def count_class_distribution(images_path):
    
    '''
        Get the number of images in each class. Call before augmenting dataset so that each class has same number of examples.
    '''

    classdict = {}

    for im in images_path:

        a = im.split('/')[-1]
        c = int(a.split('category_number_')[1].split('_')[0])
        
        if c in classdict:
            classdict[c].append(im)
        else:
            classdict[c] = [im]

    return classdict


def augment_images(images_path):

    '''
        Main function to augment all images with filenames given in images_path.
    '''
    
    print('\nAugment Images')

    # Rotate and augment first
    for xx in tqdm(images_path):

        image = io.imread(xx)

        augmentated_images = []

        # Flip image
        image_flip, success = random_horizontal_flip(image)
        if success:
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'flip_' +  xx.split('/')[-1]
            augmentated_images.append(file_name)
            io.imsave(file_name, image_flip)

        # Rotate Image
        for yy in [-3, -2, -1, 1, 2, 3]:
            image_rot = rotateImage(image, yy*5)
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'rotate' + str(yy) + '_' +  xx.split('/')[-1]
            io.imsave(file_name, image_rot)

        # RGB flip
        for order in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            r = str(order[0])
            g = str(order[1])
            b = str(order[2])
            image_rgb_flipped = rgb_flip(image, order)
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'rgbflip' + r + g + b + '_' +  xx.split('/')[-1]
            io.imsave(file_name, image_rgb_flipped)

        # Color Jitter
        for jj in range(5):
            image_jittered = color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            # print(image_jittered.shape)
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'jitter' + str(jj) + '_' +  xx.split('/')[-1]
            io.imsave(file_name, image_jittered)


def balance_classes(images_path):

    '''
        Balances classes with randomly cropped images.
    '''

    print('\nBalance Classes')

    # Balance classes with random crop (only augment classes with lesser examples to create even distribution across all classes)
    # (Both in the source and target domain)

    classdict = count_class_distribution(images_path)
    maxCount = np.max([len(classdict[c]) for c in classdict])
    print('Initial Class distribution', [len(classdict[c]) for c in classdict])

    writtenImagesCount = {c:0 for c in classdict}

    for c in tqdm(classdict):
        n_c = len(classdict[c])

        for i in range(maxCount - n_c):
            randim = classdict[c][np.random.randint(len(classdict[c]))]
            cropnum = 1
            file_name = '/'.join(randim.split('/')[:-1]) + '/' + 'randcrop_' + str(cropnum) + '_' +  randim.split('/')[-1]
            while(True):
                if os.path.exists(file_name):
                    cropnum += 1
                    file_name = '/'.join(randim.split('/')[:-1]) + '/' + 'randcrop_' + str(cropnum) + '_' +  randim.split('/')[-1]
                else:
                    image = io.imread(randim)
                    image_randcrop = random_crop(image, 224, 20)
                    io.imsave(file_name, image_randcrop)
                    writtenImagesCount[c] += 1
                    break

    print('Added Images', [writtenImagesCount[c] for c in writtenImagesCount])


def make_augmentation_dictionary(index_list_path):

    '''
        Creates a dictionary with image filenames, and the list of their augmented images filenames.
    '''

    def get_augmentations(image_filename):

        '''
            Gets the names of the augmentation files.
        '''

        xx = image_filename

        aug_files = []

        # Flip image
        file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'flip_' +  xx.split('/')[-1]
        aug_files.append(file_name)

        # Rotate Image
        for yy in [-3, -2, -1, 1, 2, 3]:
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'rotate' + str(yy) + '_' +  xx.split('/')[-1]
            aug_files.append(file_name)

        # RGB flip
        for order in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            r = str(order[0])
            g = str(order[1])
            b = str(order[2])
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'rgbflip' + r + g + b + '_' +  xx.split('/')[-1]
            aug_files.append(file_name)

        # Color Jitter
        for jj in range(5):
            file_name = '/'.join(xx.split('/')[:-1]) + '/' + 'jitter' + str(jj) + '_' +  xx.split('/')[-1]
            aug_files.append(file_name)

        return aug_files

    def get_augmentations_prefix():

        '''
            Returns the list of prefixes for augmentation images.
        '''

        aug_prefix = []

        # Flip image
        aug_prefix.append('flip')

        # Rotate Image
        for yy in [-3, -2, -1, 1, 2, 3]:
            aug_prefix.append('rotate' + str(yy))

        # RGB flip
        for order in [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
            r = str(order[0]); g = str(order[1]); b = str(order[2])
            aug_prefix.append('rgbflip' + r + g + b)

        # Color Jitter
        for jj in range(5):
            aug_prefix.append('jitter' + str(jj))
        
        # Randcrop
        aug_prefix.append('randcrop')

        return aug_prefix

    all_fils = np.load(index_list_path)
    pfix = get_augmentations_prefix()

    only_images = [a for a in all_fils if a.split('/')[-1].split('_')[0] not in pfix]
    print(len(only_images))

    print('sanity check 1')
    for f in tqdm(only_images):
        assert f.split('/')[-1].split('_')[0]=='category'
    print('correct')

    dictionary = {}
    
    print('sanity check 2')
    for f in tqdm(only_images):
        augfiles = get_augmentations(f)
        dictionary[f] = augfiles
        for fil in augfiles:
            assert fil in all_fils
    print('correct')

    return dictionary

dataset_name = 'office_31_dataset' # 'office_31_dataset'
experiments = ['DtoA']

for dataset_exp_name in tqdm(experiments):

    print('running', dataset_exp_name)

    # Rotate and augment source images
    print('Rotation augmentation on source images')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/flip*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/randcrop*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/rotate*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/rgbflip*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/jitter*.png')
    files = glob.glob('../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/*.png')
    augment_images(files)

    # Balance source classes
    print('Class balancing on source images')
    files = glob.glob('../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/*.png')
    balance_classes(files)

    # Rotate and augment target images
    print('Rotation augmentation on target images')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/flip*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/randcrop*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/rotate*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/rgbflip*.png')
    os.system('rm -rf ../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/jitter*.png')
    files = glob.glob('../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/*.png')
    augment_images(files)

    # Balance target classes
    print('Class balancing on target images')
    files = glob.glob('../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/*.png')
    balance_classes(files)

    # Save index list source images
    temp_paths_train = os.listdir('../../data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/')
    temp_paths_train = ['data/' + dataset_name + '/' + dataset_exp_name + '/source_images/train/' + x for x in temp_paths_train]
    save_path = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/source_images_index_list_train.npy'
    np.save(save_path , np.array(temp_paths_train))

    # Save index list target images
    temp_paths_train = os.listdir('../../data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/')
    temp_paths_train = ['data/' + dataset_name + '/' + dataset_exp_name + '/target_images/train/' + x for x in temp_paths_train]
    save_path = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/target_images_index_list_train.npy'
    np.save(save_path , np.array(temp_paths_train))

    # Make augmentation dictionary for source
    source_list_filename = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/source_images_index_list_train.npy'
    source_augs = make_augmentation_dictionary(source_list_filename)
    source_aug_dict_path = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/source_images_aug_dict_train.npy'
    np.save(source_aug_dict_path, source_augs)

    # Make augmentation dictionary for target
    target_list_filename = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/target_images_index_list_train.npy'
    target_augs = make_augmentation_dictionary(target_list_filename)
    target_aug_dict_path = '../../data/' + dataset_name + '/' + dataset_exp_name + '/index_lists/target_images_aug_dict_train.npy'
    np.save(target_aug_dict_path, target_augs)
