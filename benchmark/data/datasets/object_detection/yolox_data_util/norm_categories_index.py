from copy import deepcopy
import json


p = '/data/zql/datasets/coco2017/train2017/coco_ann.json'
p = '/data/zql/datasets/face_mask/WI/Medical mask/Medical mask/Medical Mask/images/coco_ann.json'
p = '/data/zql/datasets/face_mask/make_ml/images/coco_ann.json'
p = '/data/datasets/VOCdevkit/VOC2012/JPEGImages/coco_ann.json'


def ensure_index_start_from_1_and_successive(p):
    with open(p, 'r') as f:
        data = json.load(f)

    need_norm = False
    for i, c in enumerate(data['categories']):
        if i + 1 != c['id']:
            need_norm = True
            break
    if not need_norm:
        return p
        
    categories_map = {}

    new_categories = []
    for i, c in enumerate(data['categories']):
        new_categories += [deepcopy(c)]
        new_categories[-1]['id'] = i + 1
        categories_map[c['id']] = i + 1
    data['categories'] = new_categories

    new_annotations = []
    for ann in data['annotations']:
        new_annotations += [deepcopy(ann)]
        new_annotations[-1]['category_id'] = categories_map[ann['category_id']]
    data['annotations'] = new_annotations

    with open(p + '.normed', 'w') as f:
        json.dump(data, f)
    return p + '.normed'
