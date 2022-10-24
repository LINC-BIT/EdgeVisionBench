from .....exp.alg_model_manager import ABAlgModelsManager
from .....exp.exp_tracker import OnlineDATracker
from .....scenario.scenario import Scenario

import torch
from torch import nn

def train(models, alg_models_manager: ABAlgModelsManager, config, 
          exp_tracker: OnlineDATracker, scenario: Scenario, device,
          source_train_models):

    from .compute_score import compute_scores_all_target
    
    target_dataest = scenario.get_limited_target_train_dataset()
    target_bs1_loader = scenario.build_dataloader(target_dataest, 1, 1, False, False)
    
    feature_extractor = alg_models_manager.get_model(models, 'feature extractor')
    cls = alg_models_manager.get_model(models, 'classifier')
    net = nn.Sequential(feature_extractor, cls)

    num_classes_wo_target_unknown, _, num_classes_all = 18, 1, 19
    
    discriminator_p = alg_models_manager.get_model(models, 'discriminator')
    # feature_dim = 0
    ft_num_out_channels = 0
    # from torch import nn 
    for layer in cls.modules():
        if isinstance(layer, nn.Linear):
            ft_num_out_channels = layer.in_features
            break
    target_samples_low, target_samples_high = compute_scores_all_target(target_bs1_loader,
                                           feature_extractor, ft_num_out_channels, discriminator_p, net,
                                           num_classes_wo_target_unknown, 4, device)
                    

    # only_4_rotations = True
    
    from torch.utils.data import TensorDataset
    target_batch_size = min(config['batch_size'], min(target_samples_low.size(0), target_samples_high.size(0)))
    target_train_low_loader = iter(scenario.build_dataloader(TensorDataset(target_samples_low), target_batch_size,
                                                  config['num_workers'], True, None))
    target_train_high_loader = iter(scenario.build_dataloader(TensorDataset(target_samples_high), target_batch_size,
                                                  config['num_workers'], True, None))
    source_dataset = scenario.get_merged_source_dataset('train')
    source_train_loader = iter(scenario.build_dataloader(source_dataset, config['batch_size'],
                                                  config['num_workers'], True, None))

    # images,labels = get_split_dataset_info(self.folder_txt_files_saving+self.source+'_'+self.target+'_test_high.txt',self.folder_dataset)
    # ds_target_high = CustomDataset(images,labels,img_transformer=transform_target_ss_step2,returns=3,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_train_high = torch.utils.data.DataLoader(ds_target_high, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)
                            

    # images,labels = get_split_dataset_info(self.folder_txt_files+self.target+'_test.txt',self.folder_dataset)
    # ds_target = CustomDataset(images,labels,img_transformer=transform_target_ss_step2,returns=3,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_train = torch.utils.data.DataLoader(ds_target, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)
            
    # images,labels = get_split_dataset_info(self.folder_txt_files_saving+self.source+'_'+self.target+'_test_low.txt',self.folder_dataset)
    # ds_target_low = CustomDataset(images,labels,img_transformer=transform_source_ss_step2,returns=6,is_train=True,ss_classes=self.ss_classes,n_classes=self.n_classes,only_4_rotations=self.only_4_rotations,n_classes_target=self.n_classes_target)
    # target_train_low = torch.utils.data.DataLoader(ds_target_low, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, pin_memory=True, drop_last=True)
    from ...offline.ros.network import Discriminator
        # network --------------------------------------------------------------------------------------------------------------------------

    
        
    ft_num_out_channels = 0
    # from torch import nn 
    for layer in cls.modules():
        if isinstance(layer, nn.Linear):
            ft_num_out_channels = layer.in_features
            break
    # print(ft_num_out_channels)
    discriminator_p = Discriminator(ft_num_out_channels * 2, 1, 
                                        num_ssl_classes_for_each_class=4)
    discriminator_p.to(device)

    if not config['use_weight_net_first_part']:
        # if self.use_VGG:
        #     feature_extractor = VGGFc(device,model_name='vgg19')
        # else:
        #     feature_extractor = ResNetFc(device,model_name='resnet50')
        # cls = CLS(feature_extractor.output_num(), self.n_classes+1, bottle_neck_dim=256,vgg=self.use_VGG)
        # feature_extractor.to(device)        
        # cls.to(device)
        # net = nn.Sequential(feature_extractor, cls).to(device)
        # TODO: use deepcopied models before first DA
        # pass
        net = source_train_models
        feature_extractor, cls = alg_models_manager.get_model(net, 'feature extractor'), \
            alg_models_manager.get_model(net, 'classifier')
    
    # if len(target_train_low) >= len(target_train_high):
    #     length = len(target_train_low)
    # else:
    #     length = len(target_train_high)

    # max_iter = int(self.epochs_step2*length)
    max_iter = config['num_iters']
            
    # scheduler = lambda step, initial_lr : inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
    params = list(discriminator_p.parameters())              
            
    # optimizer_discriminator_p = OptimWithSheduler(optim.SGD(params, lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)
    import torch
    optimizer_discriminator_p = torch.optim.__dict__[config["optimizer"]](params, \
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
    #             param.requires_grad = False                             
    #     params_cls = list(cls.parameters())
    #     optimizer_cls = OptimWithSheduler(optim.SGD([{'params': params_cls},{'params': feature_extractor.parameters(), 'lr': (self.learning_rate/self.divison_learning_rate_backbone)}], lr=self.learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True),scheduler)

    optimizer_cls = torch.optim.__dict__[config["optimizer"]](cls.parameters(), \
                    **(config["optimizer_args"]))
    
    # k=0
    # print('\n')
    # print('Adaptation phase--------------------------------------------------------------------------------------------------------')
    # print('\n')
    ss_weight_target = config['ss_weight_target']            
    weight_class_unknown = 1 / (len(target_samples_low) * (num_classes_wo_target_unknown / (len(source_dataset) * 1.)))
    
    # while k <self.epochs_step2:
    #     print('Epoch: ',k)
    #     iteration = cycle(target_train)

    #     if len(target_train_low) > len(target_train_high):
    #         num_iterations =  len(target_train_low)
    #         num_iterations_smaller = len(target_train_high)
    #         target_train_low_iter = iter(target_train_low)
    #         target_train_high_iter = cycle(target_train_high)
    #     else:
    #         num_iterations = len(target_train_high)
    #         num_iterations_smaller = len(target_train_low)
    #         target_train_low_iter = cycle(target_train_low)
    #         target_train_high_iter = iter(target_train_high)
    import random
    import numpy as np
    import torch.nn.functional as F
    ss_classes = 4
    
    def rotate(tensor_img, rotation_i):
        return torch.from_numpy(np.rot90(tensor_img.cpu().numpy(), rotation_i, (2, 3)).copy()).float()
    
    from tqdm import tqdm
    for iter_index in tqdm(range(config['num_iters']), dynamic_ncols=True, leave=False):

        # global entropy_loss

        # TODO:
        # (im_target_entropy,_,_) = next(target_train)
        # (im_source,im_source_ss,label_source,label_source_ss,_,_) = next(target_train_low)
        # im_source, label_source = next
        # (im_target,im_target_ss,label_target_ss) = next(target_train_high)
        
        
        im_source, label_source = next(source_train_loader)
        
        rotation_i = random.randint(0, 3)
        im_source_ss = rotate(im_source, rotation_i)
        label_source_ss = F.one_hot(torch.LongTensor([rotation_i] * im_source.size(0)), ss_classes)
        # print(label_source, num_classes_all)
        label_source = F.one_hot(label_source, num_classes_all)  
        
        im_target, = next(target_train_low_loader)
        im_target_high, = next(target_train_high_loader)
        rotation_i = random.randint(0, 3)
        im_target_ss = rotate(im_target_high, rotation_i)
        label_target_ss = F.one_hot(torch.LongTensor([rotation_i] * im_target.size(0)), ss_classes)

        im_source = im_source.to(device)
        im_source_ss = im_source_ss.to(device)
        label_source = label_source.to(device)
        label_source_ss = label_source_ss.to(device)
        
        im_target = im_target.to(device)
        im_target_ss = im_target_ss.to(device)
        label_target_ss = label_target_ss.to(device)
        # im_target_entropy = im_target_entropy.to(device)

        ft1_ss = feature_extractor.forward(im_target_ss)
        ft1_original = feature_extractor.forward(im_target)
        double_input_t = torch.cat((ft1_original, ft1_ss), 1)
        ft1_ss=double_input_t

        # (_, _, _, predict_prob_source) = net.forward(im_source)
        output_prob_source = net(im_source)
        predict_prob_source = nn.Softmax(dim=-1)(output_prob_source)

        # (_ ,_, _, _) = net.forward(im_target_entropy)
        # (_, _, _, predict_prob_target) = net.forward(im_target)
        output_prob_target = net(im_target)
        predict_prob_target = nn.Softmax(dim=-1)(output_prob_target)

        p0_t, _ = discriminator_p.forward(ft1_ss)
        p0_t = nn.Softmax(dim=-1)(p0_t)

            # =========================loss function
        class_weight = np.ones((num_classes_all,), dtype=np.dtype('f'))
        class_weight[num_classes_wo_target_unknown]= weight_class_unknown * config['weight_class_unknown']
        class_weight = (torch.from_numpy(class_weight)).to(device)
        
        from ...offline.ros.loss import CrossEntropyLoss
        from .compute_score import EntropyLoss
        ce = CrossEntropyLoss(label_source, predict_prob_source,class_weight)
        entropy = EntropyLoss(predict_prob_target)
        d1_t = CrossEntropyLoss(label_target_ss,p0_t)

        # with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
        optimizer_cls.zero_grad()
        optimizer_discriminator_p.zero_grad()
        
        loss_object_class = config['cls_weight_source'] * ce
        loss_rotation = config['ss_weight_target'] * d1_t
        entropy_loss = config['entropy_weight'] * entropy

        loss = loss_object_class + loss_rotation + entropy_loss
        loss.backward()
        
        exp_tracker.add_losses({
            'task': loss_object_class,
            'rotation': loss_rotation,
            'entropy': entropy_loss
        }, iter_index)
        exp_tracker.in_each_iteration_of_each_da()
    