from curses import raw
from .data_augment import TrainTransform, ValTransform
from .datasets.coco import COCODataset
from .datasets.mosaicdetection import MosaicDetection
from .....util import HiddenPrints
import os
import json

from .norm_categories_index import ensure_index_start_from_1_and_successive


def get_default_yolox_coco_dataset(data_dir, json_file_path, img_size=416, train=True):
    if train:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=0.5,
                    hsv_prob=1.0),
                cache=False,
            )
            
        dataset = MosaicDetection(
            dataset,
            mosaic=True,
            img_size=(img_size, img_size),
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=0.5,
                hsv_prob=1.0),
            degrees=10.0,
            translate=0.1,
            mosaic_scale=(0.1, 2),
            mixup_scale=(0.5, 1.5),
            shear=2.0,
            enable_mixup=True,
            mosaic_prob=1.0,
            mixup_prob=1.0,
            only_return_xy=True
        )
        
    else:
        with HiddenPrints():
            dataset = COCODataset(
                data_dir=data_dir,
                json_file=json_file_path,
                name='',
                img_size=(img_size, img_size),
                preproc=ValTransform(legacy=False),
            )
            
    return dataset


def remap_dataset(json_file_path, ignore_classes, category_idx_map):
    if len(ignore_classes) == 0 and category_idx_map is None:
        return json_file_path
    
    hash_str = '_'.join(ignore_classes) + str(category_idx_map)
    cached_json_file_path = f'{json_file_path}.{hash(hash_str)}'
    
    # if os.path.exists(cached_json_file_path):
    #     return cached_json_file_path
    
    with open(json_file_path, 'r') as f:
        raw_ann = json.load(f)
    id_to_idx_map = {c['id']: i for i, c in enumerate(raw_ann['categories'])}
        
    ignore_classes_id = [c['id'] for c in raw_ann['categories'] if c['name'] in ignore_classes]
    raw_ann['categories'] = [c for c in raw_ann['categories'] if c['id'] not in ignore_classes_id]
    raw_ann['annotations'] = [ann for ann in raw_ann['annotations'] if ann['category_id'] not in ignore_classes_id]
    ann_img_map = {ann['image_id']: 1 for ann in raw_ann['annotations']}
    raw_ann['images'] = [img for img in raw_ann['images'] if img['id'] in ann_img_map.keys()]
    
    # print(category_idx_map, id_to_idx_map)
    # NOTE: category idx starts from 0 or 1?
    for c in raw_ann['categories']:
        # print(c)
        c['id'] = category_idx_map[id_to_idx_map[c['id']]]
    for ann in raw_ann['annotations']:
        ann['category_id'] = category_idx_map[id_to_idx_map[ann['category_id']]]
    
    with open(cached_json_file_path, 'w') as f:
        json.dump(raw_ann, f)
        
    return cached_json_file_path
