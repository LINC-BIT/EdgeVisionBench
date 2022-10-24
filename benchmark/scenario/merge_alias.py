from re import L
from typing import Dict, List
from collections import Counter

    
def grouping(bondlist):
    # reference: https://blog.csdn.net/YnagShanwen/article/details/111344386
    groups = [] 
    break1 = False    
    while bondlist:
        pair1 = bondlist.pop(0)        
        a = 11111
        b = 10000
        while b != a:
            a = b
            for atomid in pair1:
                for i,pair2 in enumerate(bondlist):            
                    if atomid in pair2:
                        pair1 = pair1 + pair2
                        bondlist.pop(i)
                        if not bondlist:
                            break1 = True
                        break
                if break1:
                    break
            b = len(pair1)
        groups.append(pair1)
    return groups


def build_semantic_class_info(classes: List[str], aliases: List[List[str]]):
    res = []
    for c in classes:
        # print(res)
        if len(aliases) == 0:
            res += [[c]]
        else:
            find_alias = False
            for alias in aliases:
                if c in alias:
                    res += [alias]
                    find_alias = True
                    break
            if not find_alias:
                res += [[c]]
    # print(classes, res)
    return res
    

def merge_the_same_meaning_classes(classes_info_of_all_datasets):
    # print(classes_info_of_all_datasets)

    semantic_classes_of_all_datasets = []
    all_aliases = []
    for classes, aliases in classes_info_of_all_datasets.values():
        all_aliases += aliases
    for classes, aliases in classes_info_of_all_datasets.values():
        semantic_classes_of_all_datasets += build_semantic_class_info(classes, all_aliases)
        
    # print(semantic_classes_of_all_datasets)
    
    grouped_classes_of_all_datasets = grouping(semantic_classes_of_all_datasets)
    # print(grouped_classes_of_all_datasets)
    
    # final_grouped_classes_of_all_datasets = [Counter(c).most_common()[0][0] for c in grouped_classes_of_all_datasets]
    # use most common class name; if the same common, use shortest class name!
    final_grouped_classes_of_all_datasets = []
    for c in grouped_classes_of_all_datasets:
        counter = Counter(c).most_common()
        max_times = counter[0][1]
        candidate_class_names = []
        for item, times in counter:
            if times < max_times:
                break
            candidate_class_names += [item]
        candidate_class_names.sort(key=lambda x: len(x))
        
        final_grouped_classes_of_all_datasets += [candidate_class_names[0]]
        
    res = {}
    res_map = {d: {} for d in classes_info_of_all_datasets.keys()}
    
    for dataset_name, (classes, _) in classes_info_of_all_datasets.items():
        final_classes = []
        for c in classes:
            for grouped_names, final_name in zip(grouped_classes_of_all_datasets, final_grouped_classes_of_all_datasets):
                if c in grouped_names:
                    final_classes += [final_name]
                    if final_name != c:
                        res_map[dataset_name][c] = final_name
                    break
        res[dataset_name] = sorted(set(final_classes), key=final_classes.index)
    
    return res, res_map


if __name__ == '__main__':
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar10_aliases = [['automobile', 'car']]
    stl10_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    final_classes_of_all_datasets, rename_map = merge_the_same_meaning_classes({
        'CIFAR10': (cifar10_classes, cifar10_aliases),
        'STL10': (stl10_classes, [])
    })

    print(final_classes_of_all_datasets, rename_map)
