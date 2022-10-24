## How to implement a dataset?

For example, we want to implement a image classification dataset.

1. create a file in corresponding directory, i.e. `benchmark/data/datasets/image_classification`

2. create a class (inherited from `benchmark.data.datasets.ab_dataset.ABDataset`), e.g. `class YourDataset(ABDataset)`

3. register your dataset with `benchmark.data.datasets.registry.dataset_register(name, classes, classes_aliases)`, which represents the name of your dataset, the classes of your dataset, and the possible aliases of the classes. Examples refer to `benchmark/data/datasets/image_classification/cifar10.py` or other files. 

   Note that the order of `classes` must match the indexes. For example, `classes` of MNIST must be `['0', '1', '2', ..., '9']`, which means 0-th class is '0', 1-st class is '1', 2-nd class is '2', ...; `['1', '2', '0', ...]` is not correct because 0-th class is not '1' and 1-st class is not '2'.

   How to get `classes` of a dataset? For PyTorch built-in dataset (CIFAR10, MNIST, ...) and general dataset build by `ImageFolder`, you can initialize it (e.g. `dataset = CIFAR10(...)`) and get its classes by `dataset.classes`. 

   ```python
   # How to get classes in CIFAR10?
   from torchvision.datasets import CIFAR10
   dataset = CIFAR10(...)
   print(dataset.classes)
   # copy this output to @dataset_register(classes=<what you copied>)
   
   # it's not recommended to dynamically get classes, e.g.:
   # this works but runs slowly!
   from torchvision.datasets import CIFAR10 as RawCIFAR10
   dataset = RawCIFAR10(...)
   
   @dataset_register(
   	name='CIFAR10',
       classes=dataset.classes
   )
   class CIFAR10(ABDataset):
       # ...
   ```

   For object detection dataset, you can read the annotation JSON file and find `categories` information in it.

4. implement abstract function `create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]])`.

   Arguments:

   - `root_dir`: the location of data
   - `split`: `train / val / test`
   - `transform`: preprocess function in `torchvision.transforms`
   - `classes`: the same value with `dataset_register.classes`
   - `ignore_classes`: **classes should be discarded. You should remove images which belong to these ignore classes.**
   - `idx_map`: **map the original class index to new class index. For example, `{0: 2}` means the index of 0-th class will be 2 instead of 0. You should implement this by modifying the stored labels in the original dataset. **

   You should do five things in this function:

   1. if no user-defined transform is passed, you should implemented the default transform
   2. create the original dataset
   3. remove ignored classes in the original dataset if there are ignored classes
   4. map the original class index to new class index if there is index map
   5. split the original dataset to train / val / test dataset. If there's no val dataset in original dataset (e.g. DomainNetReal), you should split the original dataset to train / val / test dataset. If there's already val dataset in original dataset (e.g. CIFAR10 and ImageNet), regard the original val dataset as test dataset, and split the original train dataset into train / val dataset. Details just refer to existed files.

Example (`benchmark/data/datasets/image_classification/cifar10.py`):

```python
@dataset_register(
    name='CIFAR10', 
    # means in the original CIFAR10, 0-th class is airplane, 1-st class is automobile, ...
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
    # means 'automobile' and 'car' are the same thing actually
    class_aliases=[['automobile', 'car']] 
)
class CIFAR10(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        # 1. if no user-defined transform is passed, you should implemented the default transform
        if transform is None:
            transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
        # 2. create the original dataset
        dataset = RawCIFAR10(root_dir, split != 'test', transform=transform, download=True)
        
        # 3. remove ignored classes in the original dataset if there are ignored classes
        dataset.targets = np.asarray(dataset.targets)
        if len(ignore_classes) > 0: 
            for ignore_class in ignore_classes:
                dataset.data = dataset.data[dataset.targets != classes.index(ignore_class)]
                dataset.targets = dataset.targets[dataset.targets != classes.index(ignore_class)]
        
        # 4. map the original class index to new class index if there is index map
        if idx_map is not None:
            for ti, t in enumerate(dataset.targets):
                dataset.targets[ti] = idx_map[t]
        
        # 5. split the original dataset to train / val / test dataset.
        # there is not val dataset in CIFAR10 dataset, so we split the val dataset from the train dataset.
        if split != 'test':
            dataset = train_val_split(dataset, split)
        return dataset
```

After implementing a new dataset, you can create a test file in `example` and load the dataset by `benchmark.data.dataset.get_dataset()`. Try using this dataset to ensure it works. (Example: `example/1.py`)
