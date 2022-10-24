# from data import *
# from transformations import *
# from utilities import *
# from networks import *
import numpy as np
from random import sample, random
import random
# from compute_score import compute_scores_all_target
import torch

# STEP 1 -------------------------------------------------------------------------------------

#data-------------------------------------------------------------------------------------
from .....exp.alg_model_manager import ABAlgModelsManager
from .....exp.exp_tracker import OfflineTrainTracker
from .....scenario.scenario import Scenario


def train(models, alg_models_manager: ABAlgModelsManager, config, exp_tracker: OfflineTrainTracker, scenario: Scenario, device):

    # if self.use_VGG:
    #     feature_extractor = VGGFc(self.device,model_name='vgg19')
    # else:
    #     feature_extractor = ResNetFc(self.device,model_name='resnet50')

    feature_extractor = alg_models_manager.get_model(models, 'feature extractor')
    
    source_dataset = scenario.get_merged_source_dataset('train')
    source_loader = iter(scenario.build_dataloader(source_dataset, config['batch_size'], config['num_workers'],
                                              True, None))
    
    # TODO: dataset and dataloader...
    #### source on which perform training of cls and self-sup task            
    # images,labels = get_split_dataset_info(self.folder_txt_files+self.source+'_train_all.txt',self.folder_dataset)
    # ds_source_ss = CustomDataset(images,labels,img_transformer=transform_source_ss,returns=6,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # source_train_ss = torch.utils.data.DataLoader(ds_source_ss, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)

    # images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
    # ds_target_train = CustomDataset(images,labels,img_transformer=transform_target_train,returns=2,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_train = torch.utils.data.DataLoader(ds_target_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)

    #     #### target on which compute the scores to select highest batch (integrate to the learning of ss task) and lower batch (integrate to the learning of cls task for the class unknown)
    # images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
    # ds_target_test_for_scores = CustomDataset(images,labels,img_transformer=transform_target_test_for_scores,returns=2,is_train=False,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_test_for_scores = torch.utils.data.DataLoader(ds_target_test_for_scores, batch_size=1, shuffle=False, num_workers=self.n_workers, pin_memory=True, drop_last=False)

    #### target for the final evaluation
    # images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
    # ds_target_test = CustomDataset(images,labels,img_transformer=transform_target_test,returns=2,is_train=False,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_test = torch.utils.data.DataLoader(ds_target_test, batch_size=1, shuffle=False, num_workers=self.n_workers, pin_memory=True, drop_last=False)

    # network -----------------------------------------------------------------------------------------------
    from .network import Discriminator
    
    num_classes_wo_target_unknown, _, num_classes_all = scenario.get_num_classes()
    cls = alg_models_manager.get_model(models, 'classifier')
    
    ft_num_out_channels = 0
    from torch import nn 
    for layer in cls.modules():
        if isinstance(layer, nn.Linear):
            ft_num_out_channels = layer.in_features
            break
    print(ft_num_out_channels)
    discriminator_p = Discriminator(ft_num_out_channels * 2, num_classes_wo_target_unknown, 
                                        num_ssl_classes_for_each_class=4)
    # cls = CLS(feature_extractor.output_num(), self.n_classes+1, bottle_neck_dim=256,vgg=self.use_VGG)
    

    discriminator_p.to(device)
    feature_extractor.to(device)        
    cls.to(device)                 

    from torch import nn
    net = nn.Sequential(feature_extractor, cls).to(device)
    
    from .loss import CenterLoss
    # TODO: 256?
    center_loss = CenterLoss(num_classes=4 * num_classes_wo_target_unknown, feat_dim=256 * num_classes_wo_target_unknown, 
                             use_gpu='cuda' in device, device=device)
    # if self.use_VGG:
    #     center_loss_object = CenterLoss(num_classes=self.n_classes, feat_dim=4096, use_gpu=True,device=self.device)
    # else:
    #     center_loss_object = CenterLoss(num_classes=self.n_classes, feat_dim=2048, use_gpu=True,device=self.device)
    # center_loss_object = CenterLoss(num_classes=num_classes_wo_target_unknown, feat_dim=2048, 
    #                                 use_gpu='cuda' in device, device=device)
    
    # scheduler, optimizer ---------------------------------------------------------
    # max_iter = int(self.epochs_step1*len(source_train_ss))
    # scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
                        
    params = list(discriminator_p.parameters())
    if config['weight_center_loss'] > 0:
        params = params + list(center_loss.parameters())

    # optimizer_discriminator_p = OptimWithSheduler(optim.SGD(params, lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)
    optimizer_discriminator_p = torch.optim.__dict__[config["optimizer"]](params, \
                    **(config["optimizer_args"]))
    # scheduler = torch.optim.lr_scheduler.__dict__[config['scheduler']](optimizer, **config['scheduler_args'])

    # NOTE: net must be pretrained!
    optimizer_cls = torch.optim.__dict__[config["optimizer"]](cls.parameters(), \
                    **(config["optimizer_args"]))
    
    # if not self.use_VGG:
    #     for name,param in feature_extractor.named_parameters():
    #         words= name.split('.')
    #         if words[1] =='layer4':
    #             param.requires_grad = True
    #         else:
    #             param.requires_grad = False  

    #     params_cls = list(cls.parameters())
    #     optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)   

    # else:
    #     for name,param in feature_extractor.named_parameters():
    #         words= name.split('.')
    #         if words[1] =='classifier':
    #             param.requires_grad = True
    #         else:
    # #             param.requires_grad = False                          
    # #     params_cls = list(cls.parameters())                            
    # #     optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)


    # log = Logger(self.folder_log+'/step', clear=True)
    # target_train = cycle(target_train)

    # k=0
    # print('\n')
    # print('Separation known/unknown phase------------------------------------------------------------------------------------------')
    # print('\n')

    # while k <self.epochs_step1:
    #     print('Epoch: ',k)
    #     for (i, (im_source,im_source_ss,label_source,label_source_ss,label_source_ss_center,label_source_center_object)) in enumerate(source_train_ss):
    ss_classes = 4
    
    def rotate(tensor_img, rotation_i):
        return torch.from_numpy(np.rot90(tensor_img.cpu().numpy(), rotation_i, (2, 3)).copy()).float()
    import torch.nn.functional as F

    from tqdm import tqdm
    for iter_index in tqdm(range(config['num_iters']), dynamic_ncols=True):
        im_source, label_source = next(source_loader)
        rotation_i = random.randint(0, 3)
        im_source_ss = rotate(im_source, rotation_i)
        label_source_ss = F.one_hot(ss_classes*label_source+rotation_i, ss_classes*num_classes_wo_target_unknown)
        label_source_ss_center = (4 * label_source) + rotation_i
        label_source_center_object = label_source
        label_source = F.one_hot(label_source, num_classes_all)  
        
        # im_source,im_source_ss,label_source,label_source_ss,label_source_ss_center,label_source_center_object = None
    
        # im_target, = None

        # global loss_object_class
        # global acc_train
        # global loss_rotation
        # global acc_train_rot
        # global loss_center

        im_source = im_source.to(device)
        # im_target = im_target.to(device)
        im_source_ss = im_source_ss.to(device)
        label_source = label_source.to(device)
        label_source_ss = label_source_ss.to(device)
        label_source_ss_center = label_source_ss_center.to(device)
        label_source_center_object = label_source_center_object.to(device)
        
        # print(im_source.size())

        # (_, _, _, predict_prob_source) = net.forward(im_source)
        predict_prob_source = nn.Softmax(dim=-1)(net(im_source))
        # (_, _, _, _) = net.forward(im_target)
        # net(im_target)

        fs1_ss = feature_extractor(im_source_ss)

        fs1_original = feature_extractor(im_source)
        # _ = feature_extractor.forward(im_target)

        double_input = torch.cat((fs1_original, fs1_ss), 1)
        fs1_ss = double_input  

        p0, p0_center = discriminator_p(fs1_ss) 
        p0 = nn.Softmax(dim=-1)(p0)                
        # =========================loss function
        from .loss import CrossEntropyLoss
        ce = CrossEntropyLoss(label_source, predict_prob_source)
        d1 = CrossEntropyLoss(label_source_ss,p0)
        center, _ = center_loss(p0_center, label_source_ss_center)

        # with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
        
        loss_object_class = config['cls_weight_source'] * ce
        loss_rotation = config['ss_weight_source'] * d1
        loss_center = config['weight_center_loss'] * center
        loss = loss_object_class + loss_rotation + loss_center
        
        optimizer_cls.zero_grad()
        optimizer_discriminator_p.zero_grad()
        
        loss.backward()
        
        if config['weight_center_loss'] > 0:
            for param in center_loss.parameters():
                param.grad.data *= (1. / config['weight_center_loss'])
        
        optimizer_cls.step()
        optimizer_discriminator_p.step()
        
        exp_tracker.add_losses({
            'task': loss_object_class,
            'rotation': loss_rotation,
            'center': loss_center,
            'total': loss
        }, iter_index)
        
        if iter_index % 10 == 0:
            exp_tracker.add_running_perf_status(iter_index)
        if iter_index % 500 == 0:
            met_better_model = exp_tracker.add_val_accs(iter_index)
            if met_better_model:
                alg_models_manager.set_model(models, 'discriminator', discriminator_p)
                exp_tracker.add_models()
