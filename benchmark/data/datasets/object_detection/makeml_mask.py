from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split, train_val_test_split
from typing import Dict, List, Optional
from torchvision.transforms import Compose
from .yolox_data_util.api import get_default_yolox_coco_dataset, remap_dataset, ensure_index_start_from_1_and_successive
import os

from ..registery import dataset_register


@dataset_register(
    name='MakeML_Mask', 
    classes=['face_no_mask', 'face_with_mask'], 
    task_type='Object Detection',
    object_type='Face Mask',
    class_aliases=[],
    shift_type=None
)
class MakeML_Mask(ABDataset):
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        assert transform is None, \
            'The implementation of object detection datasets is based on YOLOX (https://github.com/Megvii-BaseDetection/YOLOX) ' \
            'where normal `torchvision.transforms` is not supported. You can re-implement the dataset to override default data aug.'
        
        ann_json_file_path = os.path.join(root_dir, 'coco_ann.json')
        assert os.path.exists(ann_json_file_path), \
            f'Please put the COCO annotation JSON file in root_dir: `{root_dir}/coco_ann.json`.'

        ann_json_file_path = ensure_index_start_from_1_and_successive(ann_json_file_path)
        ann_json_file_path = remap_dataset(ann_json_file_path, ignore_classes, idx_map)
        
        dataset = get_default_yolox_coco_dataset(root_dir, ann_json_file_path, train=(split == 'train'))
        
        dataset = train_val_test_split(dataset, split)
        return dataset
