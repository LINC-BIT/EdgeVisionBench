from ..ab_dataset import ABDataset
from ..dataset_split import train_val_split, train_val_test_split
from typing import Dict, List, Optional
from torchvision.transforms import Compose
from .yolox_data_util.api import get_default_yolox_coco_dataset, remap_dataset, ensure_index_start_from_1_and_successive
import os

from ..registery import dataset_register


categories = [{"supercategory": "none", "id": 0, "name": "boat"}, {"supercategory": "none", "id": 1, "name": "person"}, {"supercategory": "none", "id": 2, "name": "car"}, {"supercategory": "none", "id": 3, "name": "bus"}, {"supercategory": "none", "id": 4, "name": "horse"}, {"supercategory": "none", "id": 5, "name": "train"}, {"supercategory": "none", "id": 6, "name": "chair"}, {"supercategory": "none", "id": 7, "name": "aeroplane"}, {"supercategory": "none", "id": 8, "name": "dog"}, {"supercategory": "none", "id": 9, "name": "pottedplant"}, {"supercategory": "none", "id": 10, "name": "motorbike"}, {"supercategory": "none", "id": 11, "name": "cat"}, {"supercategory": "none", "id": 12, "name": "bicycle"}, {"supercategory": "none", "id": 13, "name": "bird"}, {"supercategory": "none", "id": 14, "name": "bottle"}, {"supercategory": "none", "id": 15, "name": "sofa"}, {"supercategory": "none", "id": 16, "name": "diningtable"}, {"supercategory": "none", "id": 17, "name": "tvmonitor"}, {"supercategory": "none", "id": 18, "name": "sheep"}, {"supercategory": "none", "id": 19, "name": "cow"}]
classes = [i['name'] for i in categories]

@dataset_register(
    name='VOC2012', 
    classes=classes, 
    task_type='Object Detection',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VOC2012(ABDataset):
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
