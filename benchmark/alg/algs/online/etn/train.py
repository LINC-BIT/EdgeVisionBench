# from data import *
# from net import *
from cmath import isnan
import datetime
from tqdm import tqdm
# if is_in_notebook():
#     from tqdm import tqdm_notebook as tqdm
from torch import optim
# from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# cudnn.deterministic = True

import torch


# def seed_everything(seed=1234):
#     import random
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     import os
#     os.environ['PYTHONHASHSEED'] = str(seed)


# seed_everything()

# if args.misc.gpus < 1:
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     gpu_ids = []
#     output_device = torch.device('cpu')
# else:
#     gpu_ids = select_GPUs(args.misc.gpus)
#     output_device = gpu_ids[0]

# now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

# log_dir = f'{args.log.root_dir}/{now}'

# logger = SummaryWriter(log_dir)

# with open(join(log_dir, 'config.yaml'), 'w') as f:
#     f.write(yaml.dump(save_config))

# log_text = open(join(log_dir, 'log.txt'), 'w')

# model_dict = {
#     'resnet50': ResNet50Fc,
#     'vgg16': VGG16Fc
# }

from torch import nn


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-6):
    '''
    entropy for multi classification
    
    predict_prob should be size of [N, C]
    
    class_level_weight should be [1, C] or [N, C] or [C]
    
    instance_level_weight should be [N, 1] or [N]
    '''
    N, C = predict_prob.size()
    
    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'
        
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob*torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)


from .....exp.alg_model_manager import ABAlgModelsManager
from .....exp.exp_tracker import OnlineDATracker
from .....scenario.scenario import Scenario

def train(models, hparams, device, alg_models_manager: ABAlgModelsManager, exp_tracker: OnlineDATracker, scenario: Scenario):

    # class TotalNet(nn.Module):
    #     def __init__(self, ft, cls, D, aux_cls, num_source_classes):
    #         super(TotalNet, self).__init__()
    #         self.feature_extractor = ft
    #         self.classifier = cls
    #         # self.discriminator = AdversarialNetwork(256)
    #         self.discriminator = D
    #         # self.classifier_auxiliary = nn.Sequential(
    #         #     nn.Linear(256, 1024),
    #         #     nn.ReLU(inplace=True),
    #         #     nn.Dropout(0.5),
    #         #     nn.Linear(1024,1024),
    #         #     nn.ReLU(inplace=True),
    #         #     nn.Dropout(0.5),
    #         #     nn.Linear(1024, classifier_output_dim),
    #         #     TorchLeakySoftmax(classifier_output_dim)
    #         # )
    #         self.classifier_auxiliary = aux_cls

    #     def forward(self, x):
    #         f = self.feature_extractor(x)
    #         f, _, __, y = self.classifier(f)
    #         d = self.discriminator(_)
    #         y_aug, d_aug = self.classifier_auxiliary(_)
    #         return y, d, y_aug, d_aug

    # totalNet = TotalNet(
        
    # )

    # logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
    from .grl import GradientReverseModule
    
    feature_extractor = alg_models_manager.get_model(models, 'feature extractor').train(True)
    classifier = alg_models_manager.get_model(models, 'classifier').train(True)
    discriminator = nn.Sequential(
        GradientReverseModule(),
        alg_models_manager.get_model(models, 'discriminator'),
    ).train(True)
    classifier_auxiliary = alg_models_manager.get_model(models, 'auxiliary classifier').train(True)
    
    
    # if args.test.test_only:
    #     assert os.path.exists(args.test.resume_file)
    #     data = torch.load(open(args.test.resume_file, 'rb'))
    #     feature_extractor.load_state_dict(data['feature_extractor'])
    #     classifier.load_state_dict(data['classifier'])
    #     discriminator.load_state_dict(data['discriminator'])
    #     classifier_auxiliary.load_state_dict(data['classifier_auxiliary'])

    #     counter = AccuracyCounter()
    #     with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
    #         for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
    #             im = im.to(output_device)
    #             label = label.to(output_device)

    #             feature = feature_extractor.forward(im)
    #             ___, __, before_softmax, predict_prob = classifier.forward(feature)

    #             counter.addOneBatch(variable_to_numpy(predict_prob),
    #                                 variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

    #     acc_test = counter.reportAccuracy()
    #     print(f'test accuracy is {acc_test}')
    #     exit(0)


    # ===================optimizer
    # TODO: optimizer, scheduler, support user-defined optimizer and scheduler...
    optimizer_finetune = torch.optim.__dict__[hparams["optimizer"]](
        [{'params': feature_extractor.parameters(), 'lr': hparams['optimizer_args']['lr'] / 10.}], 
        **hparams["optimizer_args"]
    )
    scheduler_finetune = torch.optim.lr_scheduler.__dict__[hparams['scheduler']](optimizer_finetune, **hparams['scheduler_args'])
    
    optimizer_cls = torch.optim.__dict__[hparams["optimizer"]](classifier.parameters(), \
                    **(hparams["optimizer_args"]))
    scheduler_cls = torch.optim.lr_scheduler.__dict__[hparams['scheduler']](optimizer_cls, **hparams['scheduler_args'])
    
    optimizer_discriminator = torch.optim.__dict__[hparams["optimizer"]](discriminator.parameters(), \
                    **(hparams["optimizer_args"]))
    scheduler_discriminator = torch.optim.lr_scheduler.__dict__[hparams['scheduler']](optimizer_discriminator, **hparams['scheduler_args'])
    
    optimizer_classifier_auxiliary = torch.optim.__dict__[hparams["optimizer"]](classifier_auxiliary.parameters(), \
                    **(hparams["optimizer_args"]))
    scheduler_classifier_auxiliary = torch.optim.lr_scheduler.__dict__[hparams['scheduler']](optimizer_classifier_auxiliary, 
                                                                                             **hparams['scheduler_args'])
    
    optimizers = [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]
    schedulers = [scheduler_finetune, scheduler_cls, scheduler_discriminator, scheduler_classifier_auxiliary]
    

    # total_steps = tqdm(range(args.train.min_step), desc='global step')
    # epoch_id = 0

    # while global_step < args.train.min_step:

    #     iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    #     epoch_id += 1

    # for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):
    
    
    source_dataset = scenario.get_merged_source_dataset('train')
    source_loader = iter(scenario.build_dataloader(source_dataset, hparams['batch_size'], hparams['num_workers'], True, None))
    target_dataset = scenario.get_limited_target_train_dataset()
    target_loader = iter(scenario.build_dataloader(target_dataset, hparams['batch_size'], hparams['num_workers'], True, None))
    
    from tqdm import tqdm
    
    for iter_index in tqdm(range(hparams['num_iters']), dynamic_ncols=True, leave=False):

        # save_label_target = label_target  # for debug usage
        
        im_source, label_source = next(source_loader)
        im_target, = next(target_loader)

        label_source = label_source.to(device)

        # =========================forward pass
        im_source = im_source.to(device)
        im_target = im_target.to(device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        # TODO: we assert NO bottleneck!
        # TODO: use hooks to obtain these outputs, instead of directly return them!
        # TODO: softmax-ed predict can be computed outside instead of computing them in the internal forward()!
        feature_source = classifier.forward(fc1_s)
        feature_target = classifier.forward(fc1_t)
        predict_prob_target = nn.Softmax(dim=-1)(feature_target)

        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        predict_prob_source_aug, domain_prob_source_aug = classifier_auxiliary.forward(feature_source.detach())
        predict_prob_target_aug, domain_prob_target_aug = classifier_auxiliary.forward(feature_target.detach())

        # ==============================compute loss
        weight = (1.0 - domain_prob_source_aug)
        weight = weight / (torch.mean(weight, dim=0, keepdim=True) + 1e-10)
        weight = weight.detach()

        # fc2_s: non-softmaxed model source output 
        # ============================== cross entropy loss, it receives logits as its inputs
        ce = nn.CrossEntropyLoss(reduction='none')(feature_source, label_source).view(-1, 1)
        ce = torch.mean(ce * weight, dim=0, keepdim=True)

        tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0, keepdim=True)
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))

        def one_hot(label):
            import torch.nn.functional as F
            return torch.clamp(F.one_hot(label, num_classes=feature_source.size(1)).float(), 1., 1. - 1e-8)
        
        if torch.isnan(predict_prob_source_aug).sum() > 0 or torch.isnan(predict_prob_target_aug).sum() > 0:
            print('loss exploded!')
            return
        
        # print(predict_prob_source_aug)
        # print(domain_prob_source_aug)
        # print(predict_prob_target_aug)
        # print(domain_prob_target_aug)
        # print()
        # print(predict_prob_source_aug, domain_prob_source_aug)
        # try:
        ce_aug = nn.BCELoss(reduction='none')(torch.clamp(predict_prob_source_aug, 1e-8, 1. - 1e-8), one_hot(label_source))
        ce_aug = torch.sum(ce_aug) / label_source.numel()
        adv_loss_aug = nn.BCELoss()(torch.clamp(domain_prob_source_aug, 1e-8, 1. - 1e-8), torch.ones_like(domain_prob_source_aug))
        adv_loss_aug += nn.BCELoss()(torch.clamp(domain_prob_target_aug, 1e-8, 1. - 1e-8), torch.zeros_like(domain_prob_target_aug))
        # except:
        #     print(predict_prob_source_aug)
        #     print(domain_prob_source_aug)
        #     print(predict_prob_target_aug)
        #     print(domain_prob_target_aug)
        #     print()

        entropy = EntropyLoss(predict_prob_target)

        loss = ce + hparams['adv_loss_tradeoff'] * adv_loss + hparams['entropy_tradeoff'] * entropy + \
            hparams['adv_loss_aug_tradeoff'] * adv_loss_aug + hparams['ce_aug_tradeoff'] * ce_aug
            
        [o.zero_grad() for o in optimizers]
        loss.backward()
        [o.step() for o in optimizers]
        [s.step for s in schedulers]
            
        exp_tracker.add_losses({
            'entropy': entropy,
            'adv': adv_loss,
            'ce': ce,
            'adv_loss_aug': adv_loss_aug,
            'ce_aug': ce_aug    
        }, iter_index)
        exp_tracker.in_each_iteration_of_each_da()
