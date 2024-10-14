from typing import Dict, List, Optional, Type, Union
from benchmark.data.datasets.ab_dataset import ABDataset
#from benchmark.data.visualize import visualize_classes_in_object_detection
from benchmark.scenario.val_domain_shift import get_val_domain_shift_transform
from ..data.dataset import get_dataset, get_num_limited_dataset
import copy
from torchvision.transforms import Compose

from .merge_alias import merge_the_same_meaning_classes
from ..data.datasets.registery import static_dataset_registery


# some legacy aliases of variables:
# ignore_classes == discarded classes
# private_classes == unknown classes in partial / open-set / universal DA


def _merge_the_same_meaning_classes(classes_info_of_all_datasets):
    final_classes_of_all_datasets, rename_map = merge_the_same_meaning_classes(classes_info_of_all_datasets)
    return final_classes_of_all_datasets, rename_map


def _find_ignore_classes_when_source_a_to_target_b(a_classes: List[str], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_ignore_classes, b_ignore_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 1: # ignore extra classes in a
            a_ignore_classes = set(a_classes) - set(b_classes)
        
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 2: # ignore extra classes in b
            b_ignore_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
    
    else:
        return None # a has no intersection with b, none
    
    return a_ignore_classes, b_ignore_classes


def _find_ignore_classes_when_sources_as_to_target_b(as_classes: List[List[str]], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    from functools import reduce
    a_classes = reduce(lambda res, cur: res | set(cur), as_classes, set())
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_ignore_classes, b_ignore_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 1: # ignore extra classes in a
            a_ignore_classes = set(a_classes) - set(b_classes)
        
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3 or thres == 2: # ignore extra classes in b
            b_ignore_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_ignore_classes, b_ignore_classes = [], []
        if thres == 3:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            b_ignore_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            a_ignore_classes = set(a_classes) - (set(a_classes) & set(b_classes))
    
    else:
        return None # a has no intersection with b, none
    
    as_ignore_classes = [list(set(a_classes) & set(a_ignore_classes)) for a_classes in as_classes]
    
    return as_ignore_classes, list(b_ignore_classes)


def _find_private_classes_when_source_a_to_target_b(a_classes: List[str], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_private_classes, b_private_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_private_classes, b_private_classes = [], []
        # if thres == 2 or thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
        # if thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
            
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_private_classes, b_private_classes = [], []
        if thres == 1 or thres == 0: # ignore extra classes in b
            b_private_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_private_classes, b_private_classes = [], []
        if thres == 0:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            pass
        
    else:
        return None # a has no intersection with b, none
    
    return a_private_classes, b_private_classes


def _find_private_classes_when_sources_as_to_target_b(as_classes: List[List[str]], b_classes: List[str], da_mode):
    thres = {'da': 3, 'partial_da': 2, 'open_set_da': 1, 'universal_da': 0}[da_mode]
    
    from functools import reduce
    a_classes = reduce(lambda res, cur: res | set(cur), as_classes, set())
    
    if set(a_classes) == set(b_classes):
        # a is equal to b, normal
        # 1. no ignore classes; 2. match class idx
        a_private_classes, b_private_classes = [], []
    
    elif set(a_classes) > set(b_classes):
        # a contains b, partial
        a_private_classes, b_private_classes = [], []
        # if thres == 2 or thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
        # if thres == 0: # ignore extra classes in a
        #     a_private_classes = set(a_classes) - set(b_classes)
            
    elif set(a_classes) < set(b_classes):
        # a is contained by b, open set
        a_private_classes, b_private_classes = [], []
        if thres == 1 or thres == 0: # ignore extra classes in b
            b_private_classes = set(b_classes) - set(a_classes)
    
    elif len(set(a_classes) & set(b_classes)) > 0:
        a_private_classes, b_private_classes = [], []
        if thres == 0:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 1:
            b_private_classes = set(b_classes) - (set(a_classes) & set(b_classes))
        elif thres == 2:
            # a_private_classes = set(a_classes) - (set(a_classes) & set(b_classes))
            pass
        
    else:
        return None # a has no intersection with b, none
    
    return list(b_private_classes)


class _ABDatasetMetaInfo:
    def __init__(self, name, classes, task_type, object_type, class_aliases, shift_type):
        self.name = name
        self.classes = classes
        self.class_aliases = class_aliases
        self.shift_type = shift_type
        self.task_type = task_type
        self.object_type = object_type
        
        
def _get_dist_shift_type_when_source_a_to_target_b(a: _ABDatasetMetaInfo, b: _ABDatasetMetaInfo):
    if b.shift_type is None:
        return 'Dataset Shifts'
    
    if a.name in b.shift_type.keys():
        return b.shift_type[a.name]
    
    mid_dataset_name = list(b.shift_type.keys())[0]
    mid_dataset_meta_info = _ABDatasetMetaInfo(mid_dataset_name, *static_dataset_registery[mid_dataset_name][1:])
    
    return _get_dist_shift_type_when_source_a_to_target_b(a, mid_dataset_meta_info) + ' + ' + list(b.shift_type.values())[0]


def _handle_all_datasets(source_datasets: List[_ABDatasetMetaInfo], target_datasets: List[_ABDatasetMetaInfo], da_mode):
    
    # 1. merge the same meaning classes
    classes_info_of_all_datasets = {
        d.name: (d.classes, d.class_aliases)
        for d in source_datasets + target_datasets
    }
    final_classes_of_all_datasets, rename_map = _merge_the_same_meaning_classes(classes_info_of_all_datasets)
    all_datasets_known_classes = copy.deepcopy(final_classes_of_all_datasets)
    
    # print(all_datasets_known_classes)
    
    # 2. find ignored classes according to DA mode
    source_datasets_ignore_classes, target_datasets_ignore_classes = {d.name: [] for d in source_datasets}, \
        {d.name: [] for d in target_datasets}
    source_datasets_private_classes, target_datasets_private_classes = {d.name: [] for d in source_datasets}, \
        {d.name: [] for d in target_datasets}
    target_source_relationship_map = {td.name: {} for td in target_datasets}
    # source_target_relationship_map = {sd.name: [] for sd in source_datasets}
    
    while True:
        # why repeating this?
        # For example:
        # Under da_mode='da', after resolving a -> b, if resolving a -> c let a ignore more classes, 
        # some classes in b may become private but not ignored in this pass.
        # So next pass is necessary to ignore extra private classes in b.
        
        # Wait, something might be wrong...
        # For example:
        # Under da_mode='da', some classes are ignored when resolving a -> b, 
        # but these classes shoule not be ignored when resolving a -> c?
        
        # this means a recover process is needed
        find_new_ignored_classes = False
        find_new_private_classes = False
        
        for sd in source_datasets:
            for td in target_datasets:
                sc = all_datasets_known_classes[sd.name]
                tc = all_datasets_known_classes[td.name]
                _res = _find_ignore_classes_when_source_a_to_target_b(sc, tc, da_mode)
                if _res is None:
                    continue
                s_ignore_classes, t_ignore_classes = _res
                if len(s_ignore_classes) > 0 or len(t_ignore_classes) > 0:
                    find_new_ignored_classes = True
                
                source_datasets_ignore_classes[sd.name] += s_ignore_classes
                all_datasets_known_classes[sd.name] = [c for c in all_datasets_known_classes[sd.name] if c not in s_ignore_classes]
                    
                target_datasets_ignore_classes[td.name] += t_ignore_classes
                all_datasets_known_classes[td.name] = [c for c in all_datasets_known_classes[td.name] if c not in t_ignore_classes]
                
                _res = _find_private_classes_when_source_a_to_target_b(sc, tc, da_mode)
                s_private_classes, t_private_classes = _res
                # print(sd.name, td.name, s_ignore_classes, t_ignore_classes, s_private_classes, t_private_classes)
                if len(s_private_classes) > 0 or len(t_private_classes) > 0:
                    find_new_private_classes = True
                    
                source_datasets_private_classes[sd.name] += s_private_classes
                all_datasets_known_classes[sd.name] = [c for c in all_datasets_known_classes[sd.name] if c not in s_private_classes]
                
                target_datasets_private_classes[td.name] += t_private_classes
                all_datasets_known_classes[td.name] = [c for c in all_datasets_known_classes[td.name] if c not in t_private_classes]
                
                if sd.name not in target_source_relationship_map[td.name].keys():
                    # target_source_relationship_map[td.name] += [sd.name]
                    target_source_relationship_map[td.name][sd.name] = _get_dist_shift_type_when_source_a_to_target_b(sd, td)
                # if td.name not in source_target_relationship_map[sd.name]:
                #     source_target_relationship_map[sd.name] += [td.name]
                
        if not find_new_ignored_classes and not find_new_private_classes:
            break
        
    source_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in source_datasets_ignore_classes.items()}
    target_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_ignore_classes.items()}
    source_datasets_private_classes = {k: sorted(set(v), key=v.index) for k, v in source_datasets_private_classes.items()}
    target_datasets_private_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_private_classes.items()}
    
    # print(source_datasets_private_classes, target_datasets_private_classes)
    # 3. reparse classes idx
    # 3.1. agg all used classes
    # all_used_classes = []
    # all_datasets_private_class_idx_map = {}
    global_idx = 0
    all_used_classes_idx_map = {}
    # all_datasets_known_classes = {d: [] for d in final_classes_of_all_datasets.keys()}
    for dataset_name, classes in final_classes_of_all_datasets.items():
        ignore_classes = source_datasets_ignore_classes[dataset_name] \
            if dataset_name in source_datasets_ignore_classes.keys() else target_datasets_ignore_classes[dataset_name]
        private_classes = source_datasets_private_classes[dataset_name] \
            if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
        
        # NOTE: unknown classes in source can be not merged
        # private_classes = [] \
        #     if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
            
        for c in classes:
            if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c not in private_classes:
                all_used_classes_idx_map[c] = global_idx
                global_idx += 1
            # if c not in ignore_classes and c not in private_classes:
            #     all_datasets_known_classes[dataset_name] += [c]
    
    # dataset_private_class_idx_offset = 0
    source_private_class_idx, target_private_class_idx = None, None
    all_datasets_private_class_idx = {d: None for d in final_classes_of_all_datasets.keys()}
    for dataset_name, classes in final_classes_of_all_datasets.items():
        ignore_classes = source_datasets_ignore_classes[dataset_name] \
            if dataset_name in source_datasets_ignore_classes.keys() else target_datasets_ignore_classes[dataset_name]
        private_classes = source_datasets_private_classes[dataset_name] \
            if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
        # private_classes = [] \
        #     if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
        # for c in classes:
        #     if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c in private_classes:
        #         all_used_classes_idx_map[c] = global_idx + dataset_private_class_idx_offset
                
        if len(private_classes) > 0:
            # all_datasets_private_class_idx[dataset_name] = global_idx + dataset_private_class_idx_offset
            # dataset_private_class_idx_offset += 1
            if dataset_name in source_datasets_private_classes.keys():
                if source_private_class_idx is None:
                    source_private_class_idx = global_idx if target_private_class_idx is None else target_private_class_idx + 1
                all_datasets_private_class_idx[dataset_name] = source_private_class_idx
            else:
                if target_private_class_idx is None:
                    target_private_class_idx = global_idx if source_private_class_idx is None else source_private_class_idx + 1
                all_datasets_private_class_idx[dataset_name] = target_private_class_idx
    # all_used_classes = sorted(set(all_used_classes), key=all_used_classes.index)
    # all_used_classes_idx_map = {c: i for i, c in enumerate(all_used_classes)}
    
    # print('rename_map', rename_map)
    
    # 3.2 raw_class -> rename_map[raw_classes] -> all_used_classes_idx_map
    all_datasets_e2e_idx_map = {}
    all_datasets_e2e_class_to_idx_map = {}
    for d in source_datasets + target_datasets:
        dataset_name = d.name
        cur_e2e_idx_map = {}
        cur_e2e_class_to_idx_map = {}
        
        for raw_ci, raw_c in enumerate(d.classes):
            renamed_c = raw_c if raw_c not in rename_map[dataset_name] else rename_map[dataset_name][raw_c]
            ignore_classes = source_datasets_ignore_classes[dataset_name] \
                if dataset_name in source_datasets_ignore_classes.keys() else target_datasets_ignore_classes[dataset_name]
            if renamed_c in ignore_classes:
                continue
            
            private_classes = source_datasets_private_classes[dataset_name] \
                if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
            # private_classes = [] \
            #     if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
            if renamed_c in private_classes:
                idx = all_datasets_private_class_idx[dataset_name]
            else:
                idx = all_used_classes_idx_map[renamed_c]
            cur_e2e_idx_map[raw_ci] = idx
            cur_e2e_class_to_idx_map[raw_c] = idx
            
        all_datasets_e2e_idx_map[dataset_name] = cur_e2e_idx_map
        all_datasets_e2e_class_to_idx_map[dataset_name] = cur_e2e_class_to_idx_map
        
    all_datasets_ignore_classes = {**source_datasets_ignore_classes, **target_datasets_ignore_classes}
    all_datasets_private_classes = {**source_datasets_private_classes, **target_datasets_private_classes}
    
    classes_idx_set = []
    for d, m in all_datasets_e2e_class_to_idx_map.items():
        classes_idx_set += list(m.values())
    classes_idx_set = set(classes_idx_set)
    num_classes = len(classes_idx_set)

    return all_datasets_ignore_classes, all_datasets_private_classes, all_datasets_known_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, all_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes
        
        
def _handle_all_datasets_v2(source_datasets: List[_ABDatasetMetaInfo], target_datasets: List[_ABDatasetMetaInfo], da_mode):
    
    # 1. merge the same meaning classes
    classes_info_of_all_datasets = {
        d.name: (d.classes, d.class_aliases)
        for d in source_datasets + target_datasets
    }
    final_classes_of_all_datasets, rename_map = _merge_the_same_meaning_classes(classes_info_of_all_datasets)
    all_datasets_classes = copy.deepcopy(final_classes_of_all_datasets)
    
    # print(all_datasets_known_classes)
    
    # 2. find ignored classes according to DA mode
    # source_datasets_ignore_classes, target_datasets_ignore_classes = {d.name: [] for d in source_datasets}, \
    #     {d.name: [] for d in target_datasets}
    # source_datasets_private_classes, target_datasets_private_classes = {d.name: [] for d in source_datasets}, \
    #     {d.name: [] for d in target_datasets}
    target_source_relationship_map = {td.name: {} for td in target_datasets}
    # source_target_relationship_map = {sd.name: [] for sd in source_datasets}
    
    # 1. construct target_source_relationship_map
    for sd in source_datasets:
        for td in target_datasets:
            sc = all_datasets_classes[sd.name]
            tc = all_datasets_classes[td.name]
            
            if len(set(sc) & set(tc)) == 0:
                continue
            
            target_source_relationship_map[td.name][sd.name] = _get_dist_shift_type_when_source_a_to_target_b(sd, td)
    
    print(target_source_relationship_map)
    # exit()
    
    source_datasets_ignore_classes = {}
    for td_name, v1 in target_source_relationship_map.items():
        for sd_name, v2 in v1.items():
            source_datasets_ignore_classes[sd_name + '|' + td_name] = []
    target_datasets_ignore_classes = {d.name: [] for d in target_datasets}
    target_datasets_private_classes = {d.name: [] for d in target_datasets}
    # 保证对于每个目标域上的DA都符合给定的label shift
    # 所以不同目标域就算对应同一个源域，该源域也可能不相同
    
    for td_name, v1 in target_source_relationship_map.items():
        sd_names = list(v1.keys())
        
        sds_classes = [all_datasets_classes[sd_name] for sd_name in sd_names]
        td_classes = all_datasets_classes[td_name]

        ss_ignore_classes, t_ignore_classes = _find_ignore_classes_when_sources_as_to_target_b(sds_classes, td_classes, da_mode)
        t_private_classes = _find_private_classes_when_sources_as_to_target_b(sds_classes, td_classes, da_mode)
        
        for sd_name, s_ignore_classes in zip(sd_names, ss_ignore_classes):
            source_datasets_ignore_classes[sd_name + '|' + td_name] = s_ignore_classes
        target_datasets_ignore_classes[td_name] = t_ignore_classes
        target_datasets_private_classes[td_name] = t_private_classes

    source_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in source_datasets_ignore_classes.items()}
    target_datasets_ignore_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_ignore_classes.items()}
    target_datasets_private_classes = {k: sorted(set(v), key=v.index) for k, v in target_datasets_private_classes.items()}
    
    # for k, v in source_datasets_ignore_classes.items():
    #     print(k, len(v))
    # print()
    # for k, v in target_datasets_ignore_classes.items():
    #     print(k, len(v))
    # print()
    # for k, v in target_datasets_private_classes.items():
    #     print(k, len(v))
    # print()
    
    # print(source_datasets_private_classes, target_datasets_private_classes)
    # 3. reparse classes idx
    # 3.1. agg all used classes
    # all_used_classes = []
    # all_datasets_private_class_idx_map = {}
    
    # source_datasets_classes_idx_map = {}
    # for td_name, v1 in target_source_relationship_map.items():
    #     for sd_name, v2 in v1.items():
    #         source_datasets_classes_idx_map[sd_name + '|' + td_name] = []
    # target_datasets_classes_idx_map = {}
    
    global_idx = 0
    all_used_classes_idx_map = {}
    # all_datasets_known_classes = {d: [] for d in final_classes_of_all_datasets.keys()}
    for dataset_name, classes in all_datasets_classes.items():
        if dataset_name not in target_datasets_ignore_classes.keys():
            ignore_classes = [0] * 100000
            for sn, sic in source_datasets_ignore_classes.items():
                if sn.startswith(dataset_name):
                    if len(sic) < len(ignore_classes):
                        ignore_classes = sic
        else:
            ignore_classes = target_datasets_ignore_classes[dataset_name]
        private_classes = [] \
            if dataset_name not in target_datasets_ignore_classes.keys() else target_datasets_private_classes[dataset_name]
        
        for c in classes:
            if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c not in private_classes:
                all_used_classes_idx_map[c] = global_idx
                global_idx += 1
                
    # print(all_used_classes_idx_map)
    
    # dataset_private_class_idx_offset = 0
    target_private_class_idx = global_idx
    target_datasets_private_class_idx = {d: None for d in target_datasets_private_classes.keys()}
    
    for dataset_name, classes in final_classes_of_all_datasets.items():
        if dataset_name not in target_datasets_private_classes.keys():
            continue
        
        # ignore_classes = target_datasets_ignore_classes[dataset_name]
        private_classes = target_datasets_private_classes[dataset_name]
        # private_classes = [] \
        #     if dataset_name in source_datasets_private_classes.keys() else target_datasets_private_classes[dataset_name]
        # for c in classes:
        #     if c not in ignore_classes and c not in all_used_classes_idx_map.keys() and c in private_classes:
        #         all_used_classes_idx_map[c] = global_idx + dataset_private_class_idx_offset
                
        if len(private_classes) > 0:
            # all_datasets_private_class_idx[dataset_name] = global_idx + dataset_private_class_idx_offset
            # dataset_private_class_idx_offset += 1
            # if dataset_name in source_datasets_private_classes.keys():
            #     if source_private_class_idx is None:
            #         source_private_class_idx = global_idx if target_private_class_idx is None else target_private_class_idx + 1
            #     all_datasets_private_class_idx[dataset_name] = source_private_class_idx
            # else:
            #     if target_private_class_idx is None:
            #         target_private_class_idx = global_idx if source_private_class_idx is None else source_private_class_idx + 1
            #     all_datasets_private_class_idx[dataset_name] = target_private_class_idx
            target_datasets_private_class_idx[dataset_name] = target_private_class_idx
            target_private_class_idx += 1
            
            
    # all_used_classes = sorted(set(all_used_classes), key=all_used_classes.index)
    # all_used_classes_idx_map = {c: i for i, c in enumerate(all_used_classes)}
    
    # print('rename_map', rename_map)
    
    # 3.2 raw_class -> rename_map[raw_classes] -> all_used_classes_idx_map
    all_datasets_e2e_idx_map = {}
    all_datasets_e2e_class_to_idx_map = {}
    
    for td_name, v1 in target_source_relationship_map.items():
        sd_names = list(v1.keys())
        sds_classes = [all_datasets_classes[sd_name] for sd_name in sd_names]
        td_classes = all_datasets_classes[td_name]
        
        for sd_name, sd_classes in zip(sd_names, sds_classes):
            cur_e2e_idx_map = {}
            cur_e2e_class_to_idx_map = {}
        
            for raw_ci, raw_c in enumerate(sd_classes):
                renamed_c = raw_c if raw_c not in rename_map[dataset_name] else rename_map[dataset_name][raw_c]
                
                ignore_classes = source_datasets_ignore_classes[sd_name + '|' + td_name]
                if renamed_c in ignore_classes:
                    continue
                
                idx = all_used_classes_idx_map[renamed_c]
                
                cur_e2e_idx_map[raw_ci] = idx
                cur_e2e_class_to_idx_map[raw_c] = idx
                
            all_datasets_e2e_idx_map[sd_name + '|' + td_name] = cur_e2e_idx_map
            all_datasets_e2e_class_to_idx_map[sd_name + '|' + td_name] = cur_e2e_class_to_idx_map
            
        cur_e2e_idx_map = {}
        cur_e2e_class_to_idx_map = {}
        for raw_ci, raw_c in enumerate(td_classes):
            renamed_c = raw_c if raw_c not in rename_map[dataset_name] else rename_map[dataset_name][raw_c]
            
            ignore_classes = target_datasets_ignore_classes[td_name]
            if renamed_c in ignore_classes:
                continue
            
            if renamed_c in target_datasets_private_classes[td_name]:
                idx = target_datasets_private_class_idx[td_name]
            else:
                idx = all_used_classes_idx_map[renamed_c]
            
            cur_e2e_idx_map[raw_ci] = idx
            cur_e2e_class_to_idx_map[raw_c] = idx
            
        all_datasets_e2e_idx_map[td_name] = cur_e2e_idx_map
        all_datasets_e2e_class_to_idx_map[td_name] = cur_e2e_class_to_idx_map
        
    all_datasets_ignore_classes = {**source_datasets_ignore_classes, **target_datasets_ignore_classes}
    # all_datasets_private_classes = {**source_datasets_private_classes, **target_datasets_private_classes}
    
    classes_idx_set = []
    for d, m in all_datasets_e2e_class_to_idx_map.items():
        classes_idx_set += list(m.values())
    classes_idx_set = set(classes_idx_set)
    num_classes = len(classes_idx_set)

    return all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes


def _build_scenario_info(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str
):
    assert da_mode in ['da', 'partial_da', 'open_set_da', 'universal_da']

    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, all_datasets_private_classes, all_datasets_known_classes, \
    all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, all_datasets_private_class_idx, \
    target_source_relationship_map, rename_map, num_classes \
        = _handle_all_datasets(source_datasets_meta_info, target_datasets_meta_info, da_mode)
        
    return all_datasets_ignore_classes, all_datasets_private_classes, all_datasets_known_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, all_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes
        
        
def _build_scenario_info_v2(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str
):
    assert da_mode in ['da', 'partial_da', 'open_set_da', 'universal_da']
    #print(*static_dataset_registery['SVHN'])
    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes \
        = _handle_all_datasets_v2(source_datasets_meta_info, target_datasets_meta_info, da_mode)
        
    return all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes


def build_scenario_manually(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str,
    num_samples_in_each_target_domain: int,
    data_dirs: Dict[str, str],
    transforms: Optional[Dict[str, Compose]] = None,
    visualize_dir_path=None
):
    configs = copy.deepcopy(locals())
    
    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, all_datasets_private_classes, all_datasets_known_classes, \
    all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, all_datasets_private_class_idx, \
    target_source_relationship_map, rename_map, num_classes \
        = _build_scenario_info(source_datasets_name, target_datasets_order, da_mode)
    
    from rich.console import Console
    console = Console(width=10000)
    
    def print_obj(_o):
        # import pprint
        # s = pprint.pformat(_o, width=140, compact=True)
        console.print(_o)
    
    console.print('configs:', style='bold red')
    print_obj(configs)
    console.print('renamed classes:', style='bold red')
    print_obj(rename_map)
    console.print('discarded classes:', style='bold red')
    print_obj(all_datasets_ignore_classes)
    console.print('unknown classes:', style='bold red')
    print_obj(all_datasets_private_classes)
    console.print('class to index map:', style='bold red')
    print_obj(all_datasets_e2e_class_to_idx_map)
    console.print('index map:', style='bold red')
    print_obj(all_datasets_e2e_idx_map)
    console = Console()
    console.print('class distribution:', style='bold red')
    class_dist = {
        k: {
            '#known classes': len(all_datasets_known_classes[k]),
            '#unknown classes': len(all_datasets_private_classes[k]),
            '#discarded classes': len(all_datasets_ignore_classes[k])
        } for k in all_datasets_ignore_classes.keys()
    }
    print_obj(class_dist)
    console.print('corresponding sources of each target:', style='bold red')
    print_obj(target_source_relationship_map)
    
    # return
    
    res_source_datasets_map = {d: {split: get_dataset(d, data_dirs[d], split, getattr(transforms, d, None),
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
                                   for split in ['train', 'val', 'test']} 
                               for d in source_datasets_name}
    res_target_datasets_map = {d: {'train': get_num_limited_dataset(get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]), 
                                                                    num_samples_in_each_target_domain),
                                   'test': get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d])
                                   } 
                               for d in list(set(target_datasets_order))}
    
    val_target_datasets_order = []
    res_val_target_datasets_map = {}
    val_target_source_map = {}
    
    import random
    from .val_domain_shift import get_val_domain_shift_transform, val_domain_shifts
    val_domain_shifts = list(val_domain_shifts.keys())
    
    def random_choose(arr):
        return arr[random.randint(0, len(arr) - 1)]
    
    for val_target_domain_i in range(len(target_datasets_order)):
        datasets_name = random_choose(source_datasets_name)
        datasets = res_source_datasets_map[datasets_name]
        val_domain_shift = random_choose(val_domain_shifts)
        severity = random.randint(1, 5)
        
        test_set = datasets['test']
        # train_set_val_domain_shift_transform = get_val_domain_shift_transform(train_set.transform, 
        #                                                             val_domain_shift, 
        #                                                             severity)
        # test_set_val_domain_shift_transform: Compose = get_val_domain_shift_transform(test_set,
        #                                                             val_domain_shift,
        #                                                             severity)
        
        # NOTE: if da_mode != 'da', we must create unknown classes manually
        # and we should not modify source domain (to re-use source-trained model)
        
        # is it possible to re-use a model in different da type scenario? NO.
        
        # 1. partial_da: remove some classes in val target dataset (ok)
        # 2. open_set_da: (TODO: how can we add extra unknown classes in target domain? use data in other datasets?)
        # 3. 
        final_datasets_name = datasets_name + f' ({val_domain_shift} {severity})'
        res_val_target_datasets_map[final_datasets_name] = {
            'train': get_num_limited_dataset(get_dataset(datasets_name, data_dirs[datasets_name], 'test', None,#test_set_val_domain_shift_transform,
                                                      all_datasets_ignore_classes[datasets_name], all_datasets_e2e_idx_map[datasets_name]), 
                                                                    num_samples_in_each_target_domain),
            'test': get_dataset(datasets_name, data_dirs[datasets_name], 'test', None,#test_set_val_domain_shift_transform,
                                                      all_datasets_ignore_classes[datasets_name], all_datasets_e2e_idx_map[datasets_name])
        }
        val_target_datasets_order += [final_datasets_name]
        val_target_source_map[final_datasets_name] = { datasets_name: 'Image Corruptions' }
    
    sources_info = []
    for d in source_datasets_name:
        source_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_source_datasets_map[d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
            **class_dist[d],
            'unknown_class_idx': all_datasets_private_class_idx[d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
            'dataset': res_source_datasets_map[d]
        }
        sources_info += [source_info]
        
    targets_info = []
    for d in target_datasets_order:
        target_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_target_datasets_map[d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
            **class_dist[d],
            'unknown_class_idx': all_datasets_private_class_idx[d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
            'dataset': res_target_datasets_map[d]
        }
        targets_info += [target_info]
        
    val_targets_info = []
    for d in val_target_datasets_order:
        raw_d = list(val_target_source_map[d].keys())[0]
        val_target_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_source_datasets_map[raw_d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[raw_d].values())),
            **class_dist[raw_d],
            'unknown_class_idx': all_datasets_private_class_idx[raw_d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[raw_d],
            'dataset': res_val_target_datasets_map[d]
        }
        val_targets_info += [val_target_info]
        
    if visualize_dir_path is not None:
        import os
        from benchmark.data.visualize import visualize_classes_image_classification, visualize_classes_in_object_detection
        
        vis_func = {
            'Image Classification': visualize_classes_image_classification,
            'Object Detection': visualize_classes_in_object_detection
        }[source_datasets_meta_info[0].task_type]
        
        for source_info in sources_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'source/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
                              p, unknown_class_idx=source_info['unknown_class_idx'])
            
        for source_info in val_targets_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'val_target/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[list(val_target_source_map[name].keys())[0]],
                              p, unknown_class_idx=source_info['unknown_class_idx'])

        for source_info in targets_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'target/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
                              p, unknown_class_idx=source_info['unknown_class_idx'])
    
    from .scenario import Scenario, DatasetMetaInfo
    val_scenario_for_hp_search = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in source_datasets_name
        },
        target_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[list(val_target_source_map[d].keys())[0]].items()
                               if v != all_datasets_private_class_idx[list(val_target_source_map[d].keys())[0]]}, 
                               all_datasets_private_class_idx[list(val_target_source_map[d].keys())[0]])
            for d in sorted(set(val_target_datasets_order), key=val_target_datasets_order.index)
        },
        target_source_map=val_target_source_map,
        target_domains_order=val_target_datasets_order,
        source_datasets=res_source_datasets_map,
        target_datasets=res_val_target_datasets_map
    )
    
    test_scenario = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in source_datasets_name
        },
        target_datasets_meta_info={
            d: DatasetMetaInfo(d,
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in sorted(set(target_datasets_order), key=target_datasets_order.index)
        },
        target_source_map=target_source_relationship_map,
        target_domains_order=target_datasets_order,
        source_datasets=res_source_datasets_map,
        target_datasets=res_target_datasets_map
    )

    return val_scenario_for_hp_search, test_scenario



def build_scenario_manually_v2(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str,
    num_samples_in_each_target_domain: int,
    data_dirs: Dict[str, str],
    transforms: Optional[Dict[str, Compose]] = None,
    visualize_dir_path=None,
    offline_source_datasets_meta_info =None
):
    configs = copy.deepcopy(locals())

    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, target_datasets_private_classes, \
        all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, target_datasets_private_class_idx, \
        target_source_relationship_map, rename_map, num_classes \
        = _build_scenario_info_v2(source_datasets_name, target_datasets_order, da_mode)
    
    from rich.console import Console
    console = Console(width=10000)
    
    def print_obj(_o):
        # import pprint
        # s = pprint.pformat(_o, width=140, compact=True)
        console.print(_o)
    
    console.print('configs:', style='bold red')
    print_obj(configs)
    console.print('renamed classes:', style='bold red')
    print_obj(rename_map)
    console.print('discarded classes:', style='bold red')
    print_obj(all_datasets_ignore_classes)
    console.print('unknown classes:', style='bold red')
    print_obj(target_datasets_private_classes)
    console.print('class to index map:', style='bold red')
    print_obj(all_datasets_e2e_class_to_idx_map)
    console.print('index map:', style='bold red')
    print_obj(all_datasets_e2e_idx_map)
    console = Console()
    # console.print('class distribution:', style='bold red')
    # class_dist = {
    #     k: {
    #         '#known classes': len(all_datasets_known_classes[k]),
    #         '#unknown classes': len(all_datasets_private_classes[k]),
    #         '#discarded classes': len(all_datasets_ignore_classes[k])
    #     } for k in all_datasets_ignore_classes.keys()
    # }
    # print_obj(class_dist)
    console.print('corresponding sources of each target:', style='bold red')
    print_obj(target_source_relationship_map)
    
    # return
    
    # res_source_datasets_map = {d: {split: get_dataset(d, data_dirs[d], split, getattr(transforms, d, None),
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
    #                                for split in ['train', 'val', 'test']} 
    #                            for d in source_datasets_name}
    # res_target_datasets_map = {d: {'train': get_num_limited_dataset(get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]), 
    #                                                                 num_samples_in_each_target_domain),
    #                                'test': get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
    #                                                   all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d])
    #                                } 
    #                            for d in list(set(target_datasets_order))}
    print(source_datasets_name)
    print(all_datasets_ignore_classes.keys())
    print(all_datasets_e2e_idx_map.keys())
    print(data_dirs.keys())
    res_source_datasets_map = {d: {split: get_dataset(d.split('|')[0], data_dirs[d.replace('_', '|').split('|')[0]], split,
                                                      getattr(transforms, d.split('|')[0], None),
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
                                   for split in ['train', 'val', 'test']} 
                               for d in all_datasets_ignore_classes.keys() if d.split('|')[0] in source_datasets_name}
    
    from functools import reduce
    res_offline_train_source_datasets_map = {}
    res_offline_train_source_datasets_map_names = {}
    
    for d in source_datasets_name:
        source_dataset_with_max_num_classes = None
        
        for ed_name, ed in res_source_datasets_map.items():
            if not ed_name.startswith(d):
                continue
            
            if source_dataset_with_max_num_classes is None:
                source_dataset_with_max_num_classes = ed
                res_offline_train_source_datasets_map_names[d] = ed_name
                
            if len(ed['train'].ignore_classes) < len(source_dataset_with_max_num_classes['train'].ignore_classes):
                source_dataset_with_max_num_classes = ed
                res_offline_train_source_datasets_map_names[d] = ed_name
                
        res_offline_train_source_datasets_map[d] = source_dataset_with_max_num_classes

    res_target_datasets_map = {d: {split: get_dataset(d, data_dirs[d.split('_')[0]], split, getattr(transforms, d, None),
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d])
                                   for split in ['train', 'test']} 
                               for d in list(set(target_datasets_order))}
    
    val_target_datasets_order = []
    res_val_target_datasets_map = {}
    val_target_source_map = {}
    
    # raise NotImplementedError('TODO:')
    
    import random
    from .val_domain_shift import get_val_domain_shift_transform, val_domain_shifts
    val_domain_shifts = list(val_domain_shifts.keys())
    
    def random_choose(arr):
        return arr[random.randint(0, len(arr) - 1)]
    
    for val_target_domain_i in range(len(target_datasets_order)):
        datasets_name = random_choose(source_datasets_name)
        datasets = res_offline_train_source_datasets_map[datasets_name]
        val_domain_shift = random_choose(val_domain_shifts)
        severity = random.randint(1, 5)
        
        test_set = datasets['test']
        #print(datasets)
        #print(datasets['train'].transform)
        # train_set_val_domain_shift_transform = get_val_domain_shift_transform(train_set.transform, 
        #                                                             val_domain_shift, 
        #                                                             severity)
        # test_set_val_domain_shift_transform = get_val_domain_shift_transform(test_set.transform,
        #                                                             val_domain_shift,
        #                                                             severity)
        # test_set_val_domain_shift_transform = get_val_domain_shift_transform(test_set,
        #                                                             val_domain_shift,
        #                                                             severity)
        # NOTE: if da_mode != 'da', we must create unknown classes manually
        # and we should not modify source domain (to re-use source-trained model)
        
        # is it possible to re-use a model in different da type scenario? NO.
        
        # 1. partial_da: remove some classes in val target dataset (ok)
        # 2. open_set_da: (TODO: how can we add extra unknown classes in target domain? use data in other datasets?)
        # TODO: we MIXUP TWO IMAGES TO generate unknown classes!
        # 3. 
        final_datasets_name = datasets_name + f' ({val_domain_shift} {severity})'
        # print(res_offline_train_source_datasets_map_names[datasets_name])
        res_val_target_datasets_map[final_datasets_name] = {
            'train': get_dataset(datasets_name, data_dirs[datasets_name.split('_')[0]], 'train', None,#test_set_val_domain_shift_transform,
                                                      all_datasets_ignore_classes[res_offline_train_source_datasets_map_names[datasets_name]], 
                                                      all_datasets_e2e_idx_map[res_offline_train_source_datasets_map_names[datasets_name]]),
            'test': get_dataset(datasets_name, data_dirs[datasets_name.split('_')[0]], 'test', None,#test_set_val_domain_shift_transform,
                                                      all_datasets_ignore_classes[res_offline_train_source_datasets_map_names[datasets_name]], 
                                                      all_datasets_e2e_idx_map[res_offline_train_source_datasets_map_names[datasets_name]])
        }
        val_target_datasets_order += [final_datasets_name]
        val_target_source_map[final_datasets_name] = { datasets_name: 'Image Corruptions' }
    
    # sources_info = []
    # for d in source_datasets_name:
    #     source_info = {
    #         'name': d,
    #         '#classes (unknown classes not merged)': len(res_source_datasets_map[d]['test'].classes),
    #         '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
    #         # **class_dist[d],
    #         # 'unknown_class_idx': all_datasets_private_class_idx[d],
    #         'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
    #         'dataset': res_source_datasets_map[d]
    #     }
    #     sources_info += [source_info]
        
    # targets_info = []
    # for d in target_datasets_order:
    #     target_info = {
    #         'name': d,
    #         '#classes (unknown classes not merged)': len(res_target_datasets_map[d]['test'].classes),
    #         '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
    #         # **class_dist[d],
    #         'unknown_class_idx': target_datasets_private_class_idx[d],
    #         'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
    #         'dataset': res_target_datasets_map[d]
    #     }
    #     targets_info += [target_info]
        
    # val_targets_info = []
    # for d in val_target_datasets_order:
    #     raw_d = list(val_target_source_map[d].keys())[0]
    #     val_target_info = {
    #         'name': d,
    #         '#classes (unknown classes not merged)': len(res_source_datasets_map[raw_d]['test'].classes),
    #         '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[raw_d].values())),
    #         # **class_dist[raw_d],
    #         'unknown_class_idx': all_datasets_private_class_idx[raw_d],
    #         'class_to_idx_map': all_datasets_e2e_class_to_idx_map[raw_d],
    #         'dataset': res_val_target_datasets_map[d]
    #     }
    #     val_targets_info += [val_target_info]
        
    # if visualize_dir_path is not None:
    #     import os
    #     from benchmark.data.visualize import visualize_classes_image_classification, visualize_classes_in_object_detection
        
    #     vis_func = {
    #         'Image Classification': visualize_classes_image_classification,
    #         'Object Detection': visualize_classes_in_object_detection
    #     }[source_datasets_meta_info[0].task_type]
        
    #     for source_info in sources_info:
    #         test_dataset = source_info['dataset']['test']
    #         name = source_info['name']
    #         p = os.path.join(visualize_dir_path, f'source/{name}.png')
    #         os.makedirs(os.path.dirname(p), exist_ok=True)
    #         vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
    #                           p, unknown_class_idx=source_info['unknown_class_idx'])
            
    #     for source_info in val_targets_info:
    #         test_dataset = source_info['dataset']['test']
    #         name = source_info['name']
    #         p = os.path.join(visualize_dir_path, f'val_target/{name}.png')
    #         os.makedirs(os.path.dirname(p), exist_ok=True)
    #         vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[list(val_target_source_map[name].keys())[0]],
    #                           p, unknown_class_idx=source_info['unknown_class_idx'])

    #     for source_info in targets_info:
    #         test_dataset = source_info['dataset']['test']
    #         name = source_info['name']
    #         p = os.path.join(visualize_dir_path, f'target/{name}.png')
    #         os.makedirs(os.path.dirname(p), exist_ok=True)
    #         vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
    #                           p, unknown_class_idx=source_info['unknown_class_idx'])
    
    from .scenario import Scenario, DatasetMetaInfo
    val_online_source_datasets = {}
    for k, v in val_target_source_map.items():
        for k1 in v.keys():
            val_online_source_datasets[k1 + '|' + k] = res_offline_train_source_datasets_map[k1]

    val_scenario_for_hp_search = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[res_offline_train_source_datasets_map_names[d]].items()}, 
                               None)
            for d in source_datasets_name
        },
        source_datasets={d: res_offline_train_source_datasets_map[d] for d in source_datasets_name},

        target_datasets_meta_info={

            d: DatasetMetaInfo(d,
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[
                                   res_offline_train_source_datasets_map_names[list(val_target_source_map[d].keys())[0]]].items()},
                               None)
            for d in val_target_datasets_order

        },


        target_datasets={**val_online_source_datasets, **res_val_target_datasets_map},
        target_domains_order=val_target_datasets_order,
        target_source_map=val_target_source_map
    )
    
    test_scenario = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[res_offline_train_source_datasets_map_names[d]].items()}, 
                               None)
            for d in source_datasets_name
        },
        source_datasets={d: res_offline_train_source_datasets_map[d] for d in source_datasets_name},

        target_datasets_meta_info={

            d: DatasetMetaInfo(d,
                                        {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if k not in target_datasets_private_classes[d]},
                               target_datasets_private_class_idx[d])
                     for d in target_datasets_order
        },
        target_datasets={**res_source_datasets_map, **res_target_datasets_map},
        target_domains_order=target_datasets_order,
        target_source_map=target_source_relationship_map
    )
    
    # test_scenario = Scenario(
    #     config=configs,
    #     source_datasets_meta_info={
    #         d: DatasetMetaInfo(d, 
    #                            {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
    #                            all_datasets_private_class_idx[d])
    #         for d in source_datasets_name
    #     },
    #     target_datasets_meta_info={
    #         d: DatasetMetaInfo(d,
    #                            {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
    #                            all_datasets_private_class_idx[d])
    #         for d in sorted(set(target_datasets_order), key=target_datasets_order.index)
    #     },
    #     target_source_map=target_source_relationship_map,
    #     target_domains_order=target_datasets_order,
    #     source_datasets=res_source_datasets_map,
    #     target_datasets=res_target_datasets_map
    # )

    return val_scenario_for_hp_search, test_scenario



# for mock data
def gen_mock_data(
    source_datasets_name: List[str],
    target_datasets_order: List[str],
    da_mode: str,
    num_samples_in_each_target_domain: int,
    data_dirs: Dict[str, str],
    transforms: Optional[Dict[str, Compose]] = None,
    visualize_dir_path=None,
    mockDataResFilePath=None
):
    configs = copy.deepcopy(locals())
    
    source_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in source_datasets_name]
    target_datasets_meta_info = [_ABDatasetMetaInfo(d, *static_dataset_registery[d][1:]) for d in list(set(target_datasets_order))]

    all_datasets_ignore_classes, all_datasets_private_classes, all_datasets_known_classes, \
    all_datasets_e2e_idx_map, all_datasets_e2e_class_to_idx_map, all_datasets_private_class_idx, \
    target_source_relationship_map, rename_map, num_classes \
        = _build_scenario_info(source_datasets_name, target_datasets_order, da_mode)
    
    from rich.console import Console
    console = Console(width=10000)
    
    def print_obj(_o):
        # import pprint
        # s = pprint.pformat(_o, width=140, compact=True)
        console.print(_o)
    
    console.print('configs:', style='bold red')
    print_obj(configs)
    console.print('renamed classes:', style='bold red')
    print_obj(rename_map)
    console.print('discarded classes:', style='bold red')
    print_obj(all_datasets_ignore_classes)
    console.print('unknown classes:', style='bold red')
    print_obj(all_datasets_private_classes)
    console.print('class to index map:', style='bold red')
    print_obj(all_datasets_e2e_class_to_idx_map)
    console.print('index map:', style='bold red')
    print_obj(all_datasets_e2e_idx_map)
    console = Console()
    console.print('class distribution:', style='bold red')
    class_dist = {
        k: {
            '#known classes': len(all_datasets_known_classes[k]),
            '#unknown classes': len(all_datasets_private_classes[k]),
            '#discarded classes': len(all_datasets_ignore_classes[k])
        } for k in all_datasets_ignore_classes.keys()
    }
    print_obj(class_dist)
    console.print('corresponding sources of each target:', style='bold red')
    print_obj(target_source_relationship_map)
    
    
    # write files
    # targetSourceRelationshipMap
    # classesInEachDatasetMap
    # indexClassMap
    ignore = False
    for k, v in target_source_relationship_map.items():
        if len(v) == 0:
            ignore = True
            break
    if ignore:
        return
    
    index_class_map = {}
    for k, v in all_datasets_e2e_class_to_idx_map.items():
        for k2, v2 in v.items():
            index_class_map[k2] = v2
    index_class_map = [(k, v) for k, v in index_class_map.items()]
    index_class_map.sort(key=lambda x: x[1])
    index_class_map = {k: v for k, v in index_class_map}
    write_content = {
        'targetSourceRelationshipMap': target_source_relationship_map,
        'classesInEachDatasetMap': {
            d: {
                'knownClasses': all_datasets_known_classes[d],
                'unknownClasses': all_datasets_private_classes[d],
                'discardedClasses': []
            }
            for d in source_datasets_name + list(set(target_datasets_order))
        },
        'indexClassMap': index_class_map
    }
    
    if mockDataResFilePath is not None:
        import json
        import os
        os.makedirs(os.path.dirname(mockDataResFilePath), exist_ok=True)
        with open(os.path.join(mockDataResFilePath), 'w') as f:
            json.dump(write_content, f, indent=2)
        
    if visualize_dir_path is None:
        return
    
    # save an image of each class
    from torchvision.transforms import Compose, Lambda
    import torch
    import matplotlib.pyplot as plt
    import os 
    from PIL import Image
    import numpy as np
    
    def get_cur_transforms(d):
        m = [
            (['CIFAR10', 'MNIST', 'EMNIST', 'STL10'], Compose([])),
            (['COCO2017', 'WI_Mask', 'VOC2012', 'MakeML_Mask'], None),
            (['GTA5', 'SuperviselyPerson', 'Cityscapes', 'BaiduPerson'], 
             (
                 Compose([]), 
                 Compose([Lambda(lambda x: torch.from_numpy(np.array(x)).long())])
            )
             )
        ]
        for k, v in m:
            if d in k:
                return v
    
    for name_list in [source_datasets_name, list(set(target_datasets_order))]:
        for d in name_list:
            if d == 'UCF101' or d == 'HMDB51':
                continue
            
            save_map = {}
            dataset = get_dataset(d, data_dirs[d], 'test', get_cur_transforms(d))
            
            if d in ['CIFAR10', 'MNIST', 'EMNIST', 'STL10']:
                for img_i in range(len(dataset)):
                    # print(dataset[img_i])
                    x, y = dataset[img_i]
                    
                    # print(y)
                    # print(x)
                    # exit()
                    class_name = dataset.classes[y]
                    
                    if int(y) not in save_map.keys():
                        save_map[int(y)] = 0
                    if save_map[int(y)] == 1:
                        continue
                    save_map[int(y)] += 1
                    
                    save_p = os.path.join(visualize_dir_path, d, f'{class_name}_{save_map[int(y)]}.png')
                    os.makedirs(os.path.dirname(save_p), exist_ok=True)
                    if isinstance(x, Image.Image):
                        x.save(save_p)
                    elif isinstance(x, np.ndarray):
                        x = Image.fromarray(x)
                        x.save(save_p)
                    else:
                        print(type(x))
                        exit()
                    print(save_p)
                    
                    if sum(save_map.values()) == len(dataset.classes):
                        break
                    
            elif d in ['COCO2017', 'WI_Mask', 'VOC2012', 'MakeML_Mask']:
                for img_i in range(len(dataset)):
                    # print(dataset[img_i])
                    x, all_y, _, _ = dataset[img_i]
                    
                    # print(y)
                    # print(x)
                    # exit()
                    tt = 0
                    for each_y in all_y:
                        tt += 1
                        if tt > 1:
                            continue
                        
                        y, bbox = int(each_y[0]), each_y[1:]
                        # print(y)
                        class_name = dataset.classes[y - 1] # start from 1
                        
                        if int(y) not in save_map.keys():
                            save_map[int(y)] = 0
                        if save_map[int(y)] == 1:
                            continue
                        save_map[int(y)] += 1
                        
                        save_p = os.path.join(visualize_dir_path, d, f'{class_name}_{save_map[int(y)]}.png')
                        os.makedirs(os.path.dirname(save_p), exist_ok=True)
                        
                        from PIL import Image, ImageDraw
                        def draw_bbox(img, bbox):
                            img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
                            draw = ImageDraw.Draw(img)
                            draw.rectangle(bbox, outline=(255, 0, 0), width=6)
                            return np.array(img)

                        cur_x = draw_bbox(x, bbox)
                        
                        if isinstance(cur_x, Image.Image):
                            cur_x.save(save_p)
                        elif isinstance(cur_x, np.ndarray):
                            cur_x = Image.fromarray(cur_x)
                            cur_x.save(save_p)
                            
                            # Image.fromarray(np.uint8(x.transpose(1, 2, 0))).save(save_p + '-raw.png')
                        else:
                            print(type(cur_x))
                            exit()
                        print(save_p)
                        
                    if sum(save_map.values()) == len(dataset.classes):
                        break
                    
            elif d in ['GTA5', 'SuperviselyPerson', 'Cityscapes', 'BaiduPerson']:
                for img_i in range(len(dataset)):
                    # print(dataset[img_i])
                    x, all_y = dataset[img_i]
                    # print(x)
                    
                    y_set = set(all_y.view(-1).cpu().numpy().tolist())
                    # print(y_set)
                    # print(x)
                    # exit()
                    tt = 0
                    for each_y in y_set:
                        
                        # y, bbox = int(each_y[0]), each_y[1:]
                        y = int(each_y)
                        
                        if y >= len(dataset.classes):
                            continue
                        
                        mask = Image.new("RGBA", list(all_y.size())[::-1], (255,0,0, 128))
                        mask = np.array(mask)
                        # print(mask[0][0])
                        # exit()
                        # print(mask.shape, list(all_y.size()))
                        # mask[:, :, 0] = 255
                        # mask[:, :, 1] = 0
                        # mask[:, :, 2] = 0
                        mask[:, :, 3] = 200 * (all_y == y).numpy()
                        # if (mask[:, :, 3].sum() / 200 < 50):
                        #     continue
                        
                        mask = Image.fromarray(mask, mode='RGBA')
                        # print(y)
                        
                        # mask = (all_y == y).unsqueeze(-1).numpy()
                        # print(mask.shape)
                        print(d, y)
                        class_name = dataset.classes[y]
                        
                        if int(y) not in save_map.keys():
                            save_map[int(y)] = 0
                        if save_map[int(y)] == 1:
                            continue
                        save_map[int(y)] += 1
                        
                        save_p = os.path.join(visualize_dir_path, d, f'{class_name}_{save_map[int(y)]}.png')
                        os.makedirs(os.path.dirname(save_p), exist_ok=True)
                        
                        from PIL import Image, ImageDraw
                        def draw_mask(x, mask):
                            # img = Image.fromarray(np.uint8(img.transpose(1, 2, 0)))
                            # draw = ImageDraw.Draw(img)
                            # draw.rectangle(bbox, outline=(255, 0, 0), width=6)
                            # return np.array(img)
                            # pass
                            # print(x.size, mask.size, x.mode, mask.mode)
                            # return Image.blend(x.convert('RGBA'), mask, 0.5)
                            # return mask
                            # return Image.fromarray(x * mask)
                            res = Image.new("RGBA", mask.size)
                            res = Image.alpha_composite(res, x.convert('RGBA'))
                            res = Image.alpha_composite(res, mask)
                            return res.convert('RGB')
                        
                        cur_x = draw_mask(x, mask)
                        
                        x.save(save_p + '-raw.png')
                        
                        if isinstance(cur_x, Image.Image):
                            cur_x.save(save_p)
                        elif isinstance(cur_x, np.ndarray):
                            cur_x = Image.fromarray(cur_x)
                            cur_x.save(save_p)
                        else:
                            print(type(cur_x))
                            exit()
                        print(save_p)
                    
                        tt += 1
                        if tt > 0:
                            break
                    
                    if sum(save_map.values()) == len(dataset.classes):
                        break
            
            elif d in ['UCF101', 'HMDB51', 'IXMAS']:
                for img_i in range(len(dataset)):
                    # print(dataset[img_i])
                    x, y = dataset[img_i]
                    # print(x.shape)
                    # exit()
                    x = x[:, 0, :, :]
                    # print(x.shape)
                    # x = x.transpose(1, 2, 0)
                    print(x.shape)

                    x = torch.from_numpy(x)
                    
                    # exit()
                    
                    # print(y)
                    # print(x)
                    # exit()
                    class_name = dataset.classes[y]
                    
                    if int(y) not in save_map.keys():
                        save_map[int(y)] = 0
                    if save_map[int(y)] == 1:
                        continue
                    save_map[int(y)] += 1
                    
                    save_p = os.path.join(visualize_dir_path, d, f'{class_name}_{save_map[int(y)]}.png')
                    os.makedirs(os.path.dirname(save_p), exist_ok=True)
                    # if isinstance(x, Image.Image):
                    #     x.save(save_p)
                    # elif isinstance(x, np.ndarray):
                    #     x = Image.fromarray(x)
                    #     x.save(save_p)
                    # else:
                    #     print(type(x))
                    #     exit()
                    
                    from torchvision.utils import save_image
                    save_image(x, save_p, normalize=True)
                    
                    print(save_p)
                    
                    if sum(save_map.values()) == len(dataset.classes):
                        break
    # for d in list(set(target_datasets_order)):
    #     save_map = {}
    #     dataset = get_dataset(d, data_dirs[d], 'test', get_cur_transforms(d))
        
    #     for img_i in range(len(dataset)):
    #         x, y = dataset[img_i]
    #         # print(x)
    #         # exit()
    #         class_name = dataset.classes[y]
            
    #         if int(y) not in save_map.keys():
    #             save_map[int(y)] = 0
    #         if save_map[int(y)] == 1:
    #             continue
    #         save_map[int(y)] += 1
            
    #         save_p = os.path.join(visualize_dir_path, d, f'{class_name}_{save_map[int(y)]}.png')
    #         os.makedirs(os.path.dirname(save_p), exist_ok=True)
    #         if isinstance(x, Image.Image):
    #             x.save(save_p)
    #         elif isinstance(x, np.ndarray):
    #             x = Image.fromarray(x)
    #             x.save(save_p)
    #         else:
    #             print(type(x))
    #             exit()
    #         print(save_p)
            
    return
    
    
    res_source_datasets_map = {d: {split: get_dataset(d, data_dirs[d], split, getattr(transforms, d, None),
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]) 
                                   for split in ['train', 'val', 'test']} 
                               for d in source_datasets_name}
    res_target_datasets_map = {d: {'train': get_num_limited_dataset(get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d]), 
                                                                    num_samples_in_each_target_domain),
                                   'test': get_dataset(d, data_dirs[d], 'test', getattr(transforms, d, None), 
                                                      all_datasets_ignore_classes[d], all_datasets_e2e_idx_map[d])
                                   } 
                               for d in list(set(target_datasets_order))}
    
    val_target_datasets_order = []
    res_val_target_datasets_map = {}
    val_target_source_map = {}
    
    import random
    from .val_domain_shift import get_val_domain_shift_transform, val_domain_shifts
    val_domain_shifts = list(val_domain_shifts.keys())
    
    def random_choose(arr):
        return arr[random.randint(0, len(arr) - 1)]
    
    for val_target_domain_i in range(len(target_datasets_order)):
        datasets_name = random_choose(source_datasets_name)
        datasets = res_source_datasets_map[datasets_name]
        val_domain_shift = random_choose(val_domain_shifts)
        severity = random.randint(1, 5)
        
        test_set = datasets['test']
        # train_set_val_domain_shift_transform = get_val_domain_shift_transform(train_set.transform, 
        #                                                             val_domain_shift, 
        #                                                             severity)
        test_set_val_domain_shift_transform = get_val_domain_shift_transform(test_set.transform, 
                                                                    val_domain_shift, 
                                                                    severity)
        
        # NOTE: if da_mode != 'da', we must create unknown classes manually
        # and we should not modify source domain (to re-use source-trained model)
        
        # is it possible to re-use a model in different da type scenario? NO.
        
        # 1. partial_da: remove some classes in val target dataset (ok)
        # 2. open_set_da: (TODO: how can we add extra unknown classes in target domain? use data in other datasets?)
        # 3. 
        final_datasets_name = datasets_name + f' ({val_domain_shift} {severity})'
        res_val_target_datasets_map[final_datasets_name] = {
            'train': get_num_limited_dataset(get_dataset(datasets_name, data_dirs[datasets_name], 'test', test_set_val_domain_shift_transform, 
                                                      all_datasets_ignore_classes[datasets_name], all_datasets_e2e_idx_map[datasets_name]), 
                                                                    num_samples_in_each_target_domain),
            'test': get_dataset(datasets_name, data_dirs[datasets_name], 'test', test_set_val_domain_shift_transform, 
                                                      all_datasets_ignore_classes[datasets_name], all_datasets_e2e_idx_map[datasets_name])
        }
        val_target_datasets_order += [final_datasets_name]
        val_target_source_map[final_datasets_name] = { datasets_name: 'Image Corruptions' }
    
    sources_info = []
    for d in source_datasets_name:
        source_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_source_datasets_map[d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
            **class_dist[d],
            'unknown_class_idx': all_datasets_private_class_idx[d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
            'dataset': res_source_datasets_map[d]
        }
        sources_info += [source_info]
        
    targets_info = []
    for d in target_datasets_order:
        target_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_target_datasets_map[d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[d].values())),
            **class_dist[d],
            'unknown_class_idx': all_datasets_private_class_idx[d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[d],
            'dataset': res_target_datasets_map[d]
        }
        targets_info += [target_info]
        
    val_targets_info = []
    for d in val_target_datasets_order:
        raw_d = list(val_target_source_map[d].keys())[0]
        val_target_info = {
            'name': d,
            '#classes (unknown classes not merged)': len(res_source_datasets_map[raw_d]['test'].classes),
            '#classes (unknown classes merged to 1 class)': len(set(all_datasets_e2e_class_to_idx_map[raw_d].values())),
            **class_dist[raw_d],
            'unknown_class_idx': all_datasets_private_class_idx[raw_d],
            'class_to_idx_map': all_datasets_e2e_class_to_idx_map[raw_d],
            'dataset': res_val_target_datasets_map[d]
        }
        val_targets_info += [val_target_info]
        
    if visualize_dir_path is not None:
        import os
        from benchmark.data.visualize import visualize_classes_image_classification, visualize_classes_in_object_detection
        
        vis_func = {
            'Image Classification': visualize_classes_image_classification,
            'Object Detection': visualize_classes_in_object_detection
        }[source_datasets_meta_info[0].task_type]
        
        for source_info in sources_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'source/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
                              p, unknown_class_idx=source_info['unknown_class_idx'])
            
        for source_info in val_targets_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'val_target/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[list(val_target_source_map[name].keys())[0]],
                              p, unknown_class_idx=source_info['unknown_class_idx'])

        for source_info in targets_info:
            test_dataset = source_info['dataset']['test']
            name = source_info['name']
            p = os.path.join(visualize_dir_path, f'target/{name}.png')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            vis_func(test_dataset, source_info['class_to_idx_map'], rename_map[name],
                              p, unknown_class_idx=source_info['unknown_class_idx'])
    
    from .scenario import Scenario, DatasetMetaInfo
    val_scenario_for_hp_search = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in source_datasets_name
        },
        target_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[list(val_target_source_map[d].keys())[0]].items()
                               if v != all_datasets_private_class_idx[list(val_target_source_map[d].keys())[0]]}, 
                               all_datasets_private_class_idx[list(val_target_source_map[d].keys())[0]])
            for d in sorted(set(val_target_datasets_order), key=val_target_datasets_order.index)
        },
        target_source_map=val_target_source_map,
        target_domains_order=val_target_datasets_order,
        source_datasets=res_source_datasets_map,
        target_datasets=res_val_target_datasets_map
    )
    
    test_scenario = Scenario(
        config=configs,
        source_datasets_meta_info={
            d: DatasetMetaInfo(d, 
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in source_datasets_name
        },
        target_datasets_meta_info={
            d: DatasetMetaInfo(d,
                               {k: v for k, v in all_datasets_e2e_class_to_idx_map[d].items() if v != all_datasets_private_class_idx[d]}, 
                               all_datasets_private_class_idx[d])
            for d in sorted(set(target_datasets_order), key=target_datasets_order.index)
        },
        target_source_map=target_source_relationship_map,
        target_domains_order=target_datasets_order,
        source_datasets=res_source_datasets_map,
        target_datasets=res_target_datasets_map
    )

    return val_scenario_for_hp_search, test_scenario