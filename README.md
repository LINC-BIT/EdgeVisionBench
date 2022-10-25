# EdgeVisionBench

![Overview](https://edgevisionbench-1258477338.cos.ap-beijing.myqcloud.com/overview.png)


## Table of contents
- [EdgeVisionBench](#edgevisionbench)
  - [Table of contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
  - [2. Datasets Repository](#2-datasets-repository)
  - [3. Get Started](#3-get-started)
    - [3.1 Setup](#31-setup)
    - [3.2 Implementation](#32-implementation)
      - [3.2.1 Model](#321-model)
      - [3.2.2 Algorithm Model Manager](#322-algorithm-model-manager)
      - [3.2.3 Algorithm](#323-algorithm)
      - [3.2.4 Dataset](#324-dataset)
    - [3.3 Offline Pre-training](#33-offline-pre-training)
      - [3.3.1 Scenario](#331-scenario)
      - [3.3.2 Model](#332-model)
      - [3.3.3 Offline Pre-Training](#333-offline-pre-training)
    - [3.4 On Device Evaluation](#34-on-device-evaluation)
      - [3.4.1 Scenario](#341-scenario)
      - [3.4.2 Model](#342-model)
      - [3.4.3 Hyperparameters Search](#343-hyperparameters-search)
      - [3.4.4 Evaluation with Monitoring](#344-evaluation-with-monitoring)
      - [3.4.5 Setup User Interface](#345-setup-user-interface)
  - [4. Dependent projects](#4-dependent-projects)


## 1. Introduction

**Vision dataset repository**

The detailed list of collected datasets locates in https://crawler995.github.io/edgevisionbench-dataset/.

In EdgeVisionBench, we collect hundreds of datasets belonging to four types of edge vision applications: (1) Image classification applications aim to recognize the category of an image; (2) Object detection applications aim to detect the category and location of each object in an image; (3) Semantic segmentation applications aim to recognize the category of each pixel in an image; (4) Action recognition applications aim to recognize the category of an action in a video clip.

**Interactive demonstration**

Among the numerous possible scenarios that audience members can interact with EdgeVisionBench, we provide a list of example scenarios in the website:

- Image Classification on Raspberry Pi 

  https://crawler995.github.io/edgevisionbench-demo-pi/

- Image Classification on Jetson Nano

  https://crawler995.github.io/edgevisionbench-demo-jetson-nano/

- Object Detection on Jetson TX2

  https://crawler995.github.io/edgevisionbench-demo-jetson-tx2/

- Semantic Segmentation on Jetson Xavier NX

  https://crawler995.github.io/edgevisionbench-demo-jetson-xavier-nx/

- Action Recognition on Jetson AGX Orin

  https://crawler995.github.io/edgevisionbench-demo-jetson-agx-orin/

**Abstract**

Vision applications powered by deep neural networks (DNN) are widely deployed on edge devices and solve the tasks of incoming data streams whose label and feature distribution shift, known as domain shift. Despite their prominent presence in the real-world edge scenarios, existing benchmarks used by domain adaptation algorithms overlook evolving domain shifts in both feature and label distribution. To address this gap, we present `EdgeVisionBench`, a benchmark including over 100 vision datasets and generating different evolving domain shifts encountered by DNNs at edge. To facilitate method evaluation, we provide an open-source package that automates dataset generation, contains popular DNN models and compression techniques, and standardizes evaluations for domain adaptation techniques on edge devices.



## 2. Datasets Repository

In EdgeVisionBench, we collect hundreds of datasets belonging to four types of edge vision applications:

- **Image classification** applications distinguish different categories of object from an image. The method first takes an image as input, then extracts the image's feature via convolutional layers, and finally outputs the probability of categories via fully connected layers. A popular image classification DNN is [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), which model consists of multiple convolutional layers and pooling layers that extract the information in image. Typically, ResNet suffers from gradient vanishing (exploding) and performance degrading when the network is  deep. ResNet thus adds BatchNorm to alleviate gradient vanishing (exploding) and adds residual connection to alleviate the performance degrading. [SENet](https://ieeexplore.ieee.org/document/341010) imports channel attention to allow the network focus the more important features. In SENet, a Squeeze & Excitation Module uses the output of a block as input, produces an channel-wise importance vector, and multiplies it into the original output of the block to strengthen the important channels and weaken the unimportant channels.
- **Object detection** applications detect coordinates of the frames containing objects (e.g., people, dogs, cars) and recognize the objects. Its mainstream networks can be divided into three parts: Backbone, net and detector. Its backbone is a Darknet53 which is divided into two parts: a root convolution layer and four stages here. Its detector is the two conected convolution layers before each output. All the remaining convolution layers form the net.
- **Semantic Segmentation** applications are widely used in medical images and driverless scenes. A typical semantic segmentation DNN has an encoder-decoder structure, in which the encoder corresponds to an image classification network and the decoder varies across different semantic segmentation DNNs. For example, in fully convolutional networks (FCN), the encoder corresponds to the four stages in ResNet and the decoder contains four convolution layers.
- **Action recognition** applications recognize an object's actions in video clips, such as speaking, waving, etc. The network is divided into spatial convolutional network and temporal convolutional network, both of which use image classification networks to perform classification tasks.

A collection of the most prevalent vision datasets that can be used to generate evolving domains for any shift type, because there are natural shifts in the input feature and label space between common datasets. We expect to facilitate the development of adaptation techniques on challenging edge scenarios.

**The detailed list of collected datasets locates in https://crawler995.github.io/edgevisionbench-dataset/.**



## 3. Get Started

### 3.1 Setup
**Requirements:**

- Linux and Windows 
- Python 3.6+
- PyTorch 1.7+
- CUDA 10.2+ 

**Preparing the virtual environment:**

1. Create a conda environment and activate it.
	```shell
	conda create -n edgevisionbench python=3.7
	conda active edgevisionbench
	```
	
2. Install PyTorch 1.7+ in the [offical website](https://pytorch.org/). A NVIDIA graphics card and PyTorch with CUDA are recommended.

   ![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

3. Clone this repository and install the dependencies.

   ```bash
   git clone https://github.com/LINC-BIT/EdgeVisionBench.git
   pip install -r requirements.txt
   ```

### 3.2 Implementation

#### 3.2.1 Model

Currently the model implementation is independent with EdgeVisionBench so it can be implemented in anywhere. For example, you can copy existing implementations somewhere in your project. In the future, we consider standardize the integration of models by registering mechanism.

#### 3.2.2 Algorithm Model Manager

Different models may have different usages. For example, an image classification model can be evaluated by less than 20 lines of simple codes, while an object detection model should be evaluated by thousands of complex codes.

A model may have different usages in different algorithms. For example, in `Tent` the model is a whole, while in `CUA` the same model needs to be split into a feature extractor and a classifier.

Therefore, the critical logic of models (e.g. forward, inference, evaluation) should be decoupled with algorithm implementation, which we define in `AlgorithmModelManager`. One can implement such a manager by inheriting and implementing pre-defined abstract class, for example:

```python
from benchmark.exp.alg_model_manager import ABAlgModelsManager

class NormalSourceTrainImageClassificationManager(ABAlgModelsManager):
    def forward_to_compute_loss(self, models, x, y):
        output = self.get_model(models, 'main model')(x)
        return F.cross_entropy(output, y)
    
    def forward(self, models, x):
        return self.get_model(models, 'main model')(x)
    
    def predict(self, models, x):
        model = self.get_model(models, 'main model')
        model.eval()
        with torch.no_grad():
            return model(x)
    
    def get_accuracy(self, models, test_dataloader):
        acc = 0
        sample_num = 0
        
        model = self.get_model(models, 'main model')
        model.eval()
        device = list(model.parameters())[0].device
        
        with torch.no_grad():
            for batch_index, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                pred = F.softmax(output, dim=1).argmax(dim=1)
                correct = torch.eq(pred, y).sum().item()
                acc += correct
                sample_num += len(y)

        acc /= sample_num
        return acc
```

#### 3.2.3 Algorithm

**Offline Pre-training Algorithm**

Some techniques has their own special pre-training methods so basic pre-training models are not enough. We define an abstract class to clarify the necessary processes in an offline pre-training algorithm. All you need is to inherit and implement abstract methods in this class.

Here is an example to implement basic offline pre-training algorithm. It can be split into three stages:

1. Register the algorithm by `@algorithm_register()`

2. Verify whether the pass-in models, `alg_models_manager` and hyperparameters are legal in `verify_args()`. 

3. Define core training logic in `train()`. In the training process, datasets and data loader can be fetched in the pass-in `scenario`, and critical metrics can be tracked by the pass-in `exp_tracker`.


```python
from ....ab_algorithm import ABOfflineTrainAlgorithm
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario
from ....registery import algorithm_register

from schema import Schema
import torch
import tqdm


@algorithm_register(
    name='NormalSourceTrain',
    stage='offline',
    supported_tasks_type=['Image Classification', 'Object Detection']
)
class NormalSourceTrain(ABOfflineTrainAlgorithm):
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert 'main model' in models.keys()
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'scheduler': str,
            'scheduler_args': dict
        }).validate(hparams)
    
    def train(self, scenario: Scenario, exp_tracker: OfflineTrainTracker):
        model = self.alg_models_manager.get_model(self.models, 'main model')
        model = model.to(self.device)

        optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            model.parameters(), **self.hparams['optimizer_args'])
        scheduler = torch.optim.lr_scheduler.__dict__[
            self.hparams['scheduler']](optimizer, **self.hparams['scheduler_args'])
        
        train_sets = scenario.get_source_datasets('train')
        train_loaders = {n: iter(scenario.build_dataloader(d, self.hparams['batch_size'], 
                                                 self.hparams['num_workers'], True, True)) for n, d in train_sets.items()}
        
        exp_tracker.start_train()
        
        for iter_index in tqdm.tqdm(range(self.hparams['num_iters']), desc='iterations',
                                    leave=False, dynamic_ncols=True):
            
            losses = {}
            for train_loader_name, train_loader in train_loaders.items():
                model.train()
                self.alg_models_manager.set_model(self.models, 'main model', model)

                x, y = next(train_loader)
                x, y = x.to(self.device), y.to(self.device)
                
                task_loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, y)

                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                
                losses[train_loader_name] = task_loss
                
            exp_tracker.add_losses(losses, iter_index)
            if iter_index % 10 == 0:
                exp_tracker.add_running_perf_status(iter_index)
            
            scheduler.step()
            
            if iter_index % 500 == 0:
                met_better_model = exp_tracker.add_val_accs(iter_index)
                if met_better_model:
                    exp_tracker.add_models()
        exp_tracker.end_train()
```

**Online Model Adaptation Algorithm**

This is similar to the implementation of offline pre-training algorithm. For example:

```python
@algorithm_register(
    name='SHOT',
    stage='online',
    supported_tasks_type=['Image Classification']
)
class SHOT(ABOnlineDAAlgorithm):
    
    def verify_args(self, models, alg_models_manager, hparams: dict):
        assert all([n in models.keys() for n in ['feature extractor', 'classifier']])
        assert all([isinstance(models[n], tuple) and len(models[n]) > 0 for n in models.keys()]), 'pass the path of model file in'
        
        Schema({
            'batch_size': int,
            'num_workers': int,
            'num_iters': int,
            'optimizer': str,
            'optimizer_args': dict,
            'updating_pseudo_label_interval': int,
            'pseudo_label_task_loss_alpha': float,
            'im_loss_alpha': float
        }).validate(hparams)

    def adapt_to_target_domain(self, scenario: Scenario, exp_tracker: OnlineDATracker):
        ft = self.alg_models_manager.get_model(self.models, 'feature extractor')
        self.optimizer = torch.optim.__dict__[self.hparams['optimizer']](
            ft.parameters(), **self.hparams['optimizer_args'])
        
        target_train_set = scenario.get_limited_target_train_dataset()
        
        for iter_index in range(self.hparams['num_iters']):
            if iter_index % self.hparams['updating_pseudo_label_interval'] == 0:
                target_train_loader = scenario.build_dataloader(target_train_set, self.hparams['batch_size'], self.hparams['num_workers'],
                                                                False, False)
                target_train_set_with_pseudo_label = obtain_label(target_train_loader, 
                                                                  self.alg_models_manager.get_model(self.models, 'feature extractor'), 
                                                                  self.alg_models_manager.get_model(self.models, 'classifier'))
                target_train_loader = scenario.build_dataloader(target_train_set_with_pseudo_label, self.hparams['batch_size'], self.hparams['num_workers'],
                                                                True, True)
                target_train_loader = iter(target_train_loader)
                
            x, y = next(target_train_loader)
            x, y = x.to(self.device), y.to(self.device)
            
            self.alg_models_manager.get_model(self.models, 'feature extractor').train()
            self.alg_models_manager.get_model(self.models, 'classifier').eval()

            task_loss, im_loss = self.alg_models_manager.forward_to_compute_loss(self.models, x, y)
            loss = self.hparams['pseudo_label_task_loss_alpha'] * task_loss + self.hparams['im_loss_alpha'] * im_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            exp_tracker.add_losses({ 
                'task': self.hparams['pseudo_label_task_loss_alpha'] * task_loss, 
                'IM': self.hparams['im_loss_alpha'] * im_loss 
            }, iter_index)
            exp_tracker.in_each_iteration_of_each_da()
```

#### 3.2.4 Dataset

If you want integrate a new dataset, you can do like this. The steps are:

1. create a class (inherited from `benchmark.data.datasets.ab_dataset.ABDataset`), e.g. `class YourDataset(ABDataset)`

2. register your dataset with `benchmark.data.datasets.registry.dataset_register(name, classes, classes_aliases)`, which represents the name of your dataset, the classes of your dataset, and the possible aliases of the classes. Examples refer to `benchmark/data/datasets/image_classification/cifar10.py` or other files. 

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
   
3. implement abstract function `create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]])`.

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
   
   Example:
   
   ```python
   @dataset_register(
       name='CIFAR10', 
       # means in the original CIFAR10, 0-th class is airplane, 1-st class is automobile, ...
       classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 
       task_type='Image Classification',
       # means 'automobile' and 'car' are the same thing actually
       class_aliases=[['automobile', 'car']],
       shift_type=None
   )
   class CIFAR10(ABDataset):    
       def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                          classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
           # 1. if no user-defined transform is passed, you should implemented the default transform
           if transform is None:
               transform = cifar_like_image_train_aug() if split == 'train' else cifar_like_image_test_aug()
               self.transform = transform
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
   


### 3.3 Offline Pre-training

Before online evaluation in edge devices, a pre-trained model is necessary. And some techniques has their own special training methods so basic pre-training models are not enough.

#### 3.3.1 Scenario

Register a scenario like:

```python
from benchmark.scenario.registery import scenario_register

num_classes = scenario_register('Image Classification (32*32)', dict(
    source_datasets_name=['CIFAR10', 'SVHN'],
    target_datasets_order=['STL10', 'MNIST', 'STL10', 'USPS', 'MNIST', 'STL10'],
    da_mode='da',
    num_samples_in_each_target_domain=100,
    data_dirs={...}
))
```

Arguments:

- `source_datasets_name`: a unordered list representing the datasets in the source domain. All datasets will be merged into a source domain.
- `target_datasets_order`: a ordered list representing the datasets in evolving target domains. Each dataset will be a target domain.
- `da_mode`: the type of shifts in label space: `'da' | 'partial_da' | 'open_set_da' | 'universal_da'`
- `num_samples_in_each_target_domain`: the number of available samples in each target domain. It will be deprecated and be replaced with `domain_interval` which limits the training time in each target domain.
- `data_dirs`: locations of data of each relevant dataset.

Return:

- `num_classes`: the number of object categories (considering unknown objects as a new category). It can be used to initialize a model like `model = resnet18(num_classes=num_classes)`.

#### 3.3.2 Model

An algorithm may need multiple models (for example, at least two models are needed in adversarial training). So you should define your models as a dict with readable information, for example:

```python
models = {
    'feature extractor': (feature_extractor_model, 'pretrained feature extractor from resnet-18'),
    'classifier': (classifier_model, 'random inited classifier'),
}
```

#### 3.3.3 Offline Pre-Training

Finally, we can run the offline pre-training like:

```python
from benchmark.exp.framework import offline_train
import time
offline_train(
    alg_name='NormalSourceTrain', 
    scenario_name='Image Classification (32*32)',
    models=models,
    alg_models_manager=NormalSourceTrainImageClassificationManager(),
    hparams={
        'batch_size': 128,
        'num_workers': 4,
        'num_iters': 80000,
        'optimizer': 'SGD',
        'optimizer_args': {
            'lr': 1e-1,
            'momentum': 0.9,
            'weight_decay': 5e-4    
        },
        'scheduler': 'MultiStepLR',
        'scheduler_args': {
            'milestones': [25000, 50000, 70000],
            'gamma': 0.2
        }
    },
    device='cuda',
    random_seed=2022,
    res_save_dir=f'./example/exp/offline/logs/{time.time()}'
)
```

### 3.4 On Device Evaluation

#### 3.4.1 Scenario

The scenario registered in the offline pre-training can be re-used in this stage.

#### 3.4.2 Model

You should load pretrained models as a dict like:

```python
models = {
    'feature extractor': (pretrained_feature_extractor_model, 'pretrained feature extractor from xxx.pth'),
    'classifier': (pretrained_classifier_model, 'pretrained classifier from xxx.pth'),
}
```

#### 3.4.3 Hyperparameters Search

After above preparations, we can run the evaluation now. However, we claim that the hyperparameters are critical in a benchmark. So before real evaluation, we should first search the hyperparameters like:

```python
offline_da_hp_search(
    alg_name='Tent', 
    scenario_name='Image Classification (32*32)',
    models=models,
    alg_models_manager=TentImageClassificationManager(),
    
    base_hparams={
        'batch_size': 100,
        'num_workers': 8,
        'num_iters': 1,
        'optimizer': 'SGD',
        'optimizer_args': {
            'lr': 1e-3,
            'momentum': 0.9
        }
    },
    hparams_search_range={
        'optimizer_args.lr': [1e-3, 5e-4, 2e-4, 1e-4],
        'num_iters': [1, 2, 5, 15, 25],
        'batch_size': [4, 8, 16, 32, 64, 100]
    },
    random_search_times=20,
    legal_hparams_condition=lambda h: h['num_iters'] * h['batch_size'] >= 100,
    reasonable_da_acc_improvement=0.01,
    
    device='cuda',
    random_seed=2022,
    res_save_dir=f'./example/exp/online/logs/{time.time()}'
)

```

This process conduct n-times random search in the given hyperparameter search space on the valiation scenario. The validation scenario is automatically generated by apply validating image corruptions on source datasets so no data leakage happens.

#### 3.4.4 Evaluation with Monitoring

Finally, we can run the final evaluation using the searched best hyperparameters like:

```python
online_da(
    alg_name='Tent', 
    scenario_name='Image Classification (32*32)',
    models=models,
    alg_models_manager=TentImageClassificationManager(),
    hparams=best_hparams,
    device='cuda',
    random_seed=2022,
    res_save_dir=f'./example/exp/online/logs/{time.time()}'
)
```

During the evaluation, the monitoring user interface can be launched by `tensorboard --logdir=<res_save_dir>/tb_log`.


#### 3.4.5 Setup User Interface

The setup UI is developed based on `React` and this project depends on `Node.js`. Before the first launch, install `Node.js` in its [official website](https://nodejs.org/en/) and install dependencies by:

```bash
cd benchmark/ui
npm i
```

Launch the setup UI by:

```python
npm start
```

Then the user interface will be automatically launched in your browser.


## 4. Dependent projects

Thanks to these projects!

- [microsoft/nni](https://github.com/microsoft/nni)
- [ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [fregu856/deeplabv3](https://github.com/fregu856/deeplabv3)
- [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed)
