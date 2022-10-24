import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import network
from . import loss
# import pre_process as prep
import torch.utils.data as util_data
# import lr_schedule
# import data_list
# from data_list import ImageList
from torch.autograd import Variable
import random
from .grl import AdversarialLayer

# optim_dict = {"SGD": optim.SGD}

# def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
#     start_test = True
#     if test_10crop:
#         iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
#         for i in range(len(loader['test0'])):
#             data = [iter_test[j].next() for j in range(10)]
#             inputs = [data[j][0] for j in range(10)]
#             labels = data[0][1]
#             if gpu:
#                 for j in range(10):
#                     inputs[j] = Variable(inputs[j].cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 for j in range(10):
#                     inputs[j] = Variable(inputs[j])
#                 labels = Variable(labels)
#             outputs = []
#             for j in range(10):
#                 _, predict_out = model(inputs[j])
#                 outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
#             softmax_outputs = sum(outputs)
#             if start_test:
#                 all_softmax_output = softmax_outputs.data.cpu().float()
#                 start_test = False
#             else:
#                 all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
#     else:
#         iter_val = iter(loader["test"])
#         for i in range(len(loader['test'])):
#             data = iter_val.next()
#             inputs = data[0]
#             if gpu:
#                 inputs = Variable(inputs.cuda())
#             else:
#                 inputs = Variable(inputs)
#             _, outputs = model(inputs)
#             softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
#             if start_test:
#                 all_softmax_output = softmax_outputs.data.cpu().float()
#                 start_test = False
#             else:
#                 all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
#     return all_softmax_output

# def image_classification_test(loader, model, test_10crop=True, gpu=True, iter_num=-1):
#     start_test = True
#     if test_10crop:
#         iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
#         for i in range(len(loader['test0'])):
#             data = [iter_test[j].next() for j in range(10)]
#             inputs = [data[j][0] for j in range(10)]
#             labels = data[0][1]
#             if gpu:
#                 for j in range(10):
#                     inputs[j] = Variable(inputs[j].cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 for j in range(10):
#                     inputs[j] = Variable(inputs[j])
#                 labels = Variable(labels)
#             outputs = []
#             for j in range(10):
#                 _, predict_out = model(inputs[j])
#                 outputs.append(nn.Softmax(dim=1)(predict_out))
#             outputs = sum(outputs)
#             if start_test:
#                 all_output = outputs.data.float()
#                 all_label = labels.data.float()
#                 start_test = False
#             else:
#                 all_output = torch.cat((all_output, outputs.data.float()), 0)
#                 all_label = torch.cat((all_label, labels.data.float()), 0)
#     else:
#         iter_test = iter(loader["test"])
#         for i in range(len(loader['test'])):
#             data = iter_test.next()
#             inputs = data[0]
#             labels = data[1]
#             if gpu:
#                 inputs = Variable(inputs.cuda())
#                 labels = Variable(labels.cuda())
#             else:
#                 inputs = Variable(inputs)
#                 labels = Variable(labels)
#             _, outputs = model(inputs)
#             if start_test:
#                 all_output = outputs.data.float()
#                 all_label = labels.data.float()
#                 start_test = False
#             else:
#                 all_output = torch.cat((all_output, outputs.data.float()), 0)
#                 all_label = torch.cat((all_label, labels.data.float()), 0)       
#     _, predict = torch.max(all_output, 1)
#     accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
#     return accuracy


def get_avg_output_in_target(model, finite_target_loader, softmax_param, device):
    # device = None
    model.eval()
    start_test = True
    with torch.no_grad():
        for x, in finite_target_loader:
            x = x.to(device)
            outputs = model(x)
            
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    return all_softmax_output


from .....exp.alg_model_manager import ABAlgModelsManager
from .....exp.exp_tracker import OnlineDATracker
from .....scenario.scenario import Scenario

def train(models, alg_models_manager: ABAlgModelsManager, config, exp_tracker: OnlineDATracker, scenario: Scenario, device):
    ## set pre-process
    # TODO: preprocess (not necessary in our benchmark because preprocess is defined in the ABDataset)
    # prep_dict = {}
    # prep_config = config["prep"]
    # prep_dict["source"] = prep.image_train( \
    #                         resize_size=prep_config["resize_size"], \
    #                         crop_size=prep_config["crop_size"])
    # prep_dict["target"] = prep.image_train( \
    #                         resize_size=prep_config["resize_size"], \
    #                         crop_size=prep_config["crop_size"])
    # if prep_config["test_10crop"]:
    #     prep_dict["test"] = prep.image_test_10crop( \
    #                         resize_size=prep_config["resize_size"], \
    #                         crop_size=prep_config["crop_size"])
    # else:
    #     prep_dict["test"] = prep.image_test( \
    #                         resize_size=prep_config["resize_size"], \
    #                         crop_size=prep_config["crop_size"])
               
    ## set loss
    # NOTE: PADA only supports Image Classification!
    # the task must satisfy that one sample matches one class!
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.PADA
    loss_params = config["loss"]

    ## prepare data
    dsets = {
        'source': scenario.get_merged_source_dataset('train'),
        'target': scenario.get_limited_target_train_dataset()
    }
    dset_loaders = {
        'source': iter(scenario.build_dataloader(dsets["source"], config['batch_size'], config['num_workers'], True, None)),
        'target': iter(scenario.build_dataloader(dsets["target"], config['batch_size'], config['num_workers'], True, None)),
        'target_finite': scenario.build_dataloader(dsets["target"], config['batch_size'], config['num_workers'], False, False),
    }

    class_num = config["num_classes"]

    ## set base network
    # TODO: prepare base_network (main model, for inference)
    # net_config = config["network"]
    # base_network = net_config["name"](**net_config["params"])
    base_network_ft = alg_models_manager.get_model(models, 'feature extractor of base model')
    base_network_cls = alg_models_manager.get_model(models, 'classifier of base model')
    # print(base_network_ft, base_network_cls)
    base_network = nn.Sequential(base_network_ft, base_network_cls)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    # TODO: prepare parameter_list, set learning rate
    # config is too messy..
    if config["fc_lr"]:
        parameter_list = [{"params":base_network_ft.parameters()}, \
                        {"params":base_network_cls.parameters(), "lr":config['fc_lr']}]
    else:
        parameter_list = [{"params":base_network.parameters()}]

    ## add additional network for some methods
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
        
    # TODO: prepare ad network and GRL, set learning rate
    # ad_net = network.AdversarialNetwork(base_network.output_num())
    # gradient_reverse_layer = network.AdversarialLayer(high_value=config["high"])
    # if use_gpu:
    #     ad_net = ad_net.cuda()
    # parameter_list.append({"params":ad_net.parameters(), "lr":10})
    ad_net = alg_models_manager.get_model(models, 'ad model')
    gradient_reverse_layer = AdversarialLayer(config['high'])
    parameter_list.append({"params":ad_net.parameters(), "lr":10})
 
    ## set optimizer
    # TODO: prepare optimizer, scheduler...
    # TODO: set scheduler hparams
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.__dict__[config["optimizer"]](parameter_list, \
                    **(config["optimizer_args"]))
    scheduler = torch.optim.lr_scheduler.__dict__[config['scheduler']](optimizer, **config['scheduler_args'])

    ## train   
    # TODO: main training loop
    # len_train_source = len(dset_loaders["source"]) - 1
    # len_train_target = len(dset_loaders["target"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    from tqdm import tqdm
    for i in tqdm(range(config["num_iterations"]), dynamic_ncols=True, leave=False, total=config['num_iterations']):
        # if i % config["test_interval"] == 0:
        #     base_network.train(False)
        #     # temp_acc = image_classification_test(dset_loaders, \
        #     #     base_network, test_10crop=prep_config["test_10crop"], \
        #     #     gpu=use_gpu)
        #     # TODO: get acc
        #     temp_acc = exp_tracker.
        #     # temp_model = nn.Sequential(base_network) # ?
        #     temp_model = base_network
        #     if temp_acc > best_acc:
        #         best_acc = temp_acc
        #         best_model = temp_model
        #     log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
        #     config["out_file"].write(log_str)
        #     config["out_file"].flush()
        #     print(log_str)
        # if i % config["snapshot_interval"] == 0:
        #     torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
        #         "iter_{:05d}_model.pth.tar".format(i)))
                    
       
        if i % loss_params["update_iter"] == loss_params["update_iter"] - 1:
            base_network.train(False)
            # TODO: ...
            # target_fc8_out = image_classification_predict(dset_loaders, base_network, softmax_param=config["softmax_param"])
            target_fc8_out = get_avg_output_in_target(base_network, dset_loaders["target_finite"], config['softmax_param'], device)
            class_weight = torch.mean(target_fc8_out, 0)
            class_weight = (class_weight / torch.mean(class_weight)).cuda().view(-1)
            class_criterion = nn.CrossEntropyLoss(weight = class_weight)
        

        ## train one iter
        base_network.train(True)
        # TODO: init optimizer in the beginning
        # optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # if i % len_train_source == 0:
        #     iter_source = iter(dset_loaders["source"])
        # if i % len_train_target == 0:
        #     iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = next(dset_loaders['source'])
        inputs_target,  = next(dset_loaders['target'])
        # if use_gpu:
        #     inputs_source, inputs_target, labels_source = \
        #         Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
        #         Variable(labels_source).cuda()
        # else:
        #     inputs_source, inputs_target, labels_source = Variable(inputs_source), \
        #         Variable(inputs_target), Variable(labels_source)
        
        inputs_source, inputs_target, labels_source = inputs_source.to(device), \
            inputs_target.to(device), labels_source.to(device)
        
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        # features, outputs = base_network(inputs)
        features = base_network_ft(inputs)
        outputs = base_network_cls(features)

        # softmax_out = nn.Softmax(dim=1)(outputs).detach() # unused variable
        
        ad_net.train(True)
        weight_ad = torch.zeros(inputs.size(0))
        label_numpy = labels_source.data.cpu().numpy()
        
        for j in range(inputs.size(0) // 2):
            weight_ad[j] = class_weight[int(label_numpy[j])]
        weight_ad = weight_ad / torch.max(weight_ad[0: inputs.size(0) // 2])
        for j in range(inputs.size(0) // 2, inputs.size(0)):
            weight_ad[j] = 1.0            
        transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
                                           weight_ad, use_gpu)

        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0) // 2), labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        exp_tracker.add_losses({
            'transfer': loss_params["trade_off"] * transfer_loss,
            'classifier': classifier_loss
        }, i)
        exp_tracker.add_histogram('running/classes_weight', weight_ad, i)
        exp_tracker.in_each_iteration_of_each_da()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Transfer Learning')
#     parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
#     parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
#     parser.add_argument('--dset', type=str, default='office', help="The dataset or source dataset used")
#     parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon_31_list.txt', help="The source dataset path list")
#     parser.add_argument('--t_dset_path', type=str, default='../data/office/webcam_10_list.txt', help="The target dataset path list")
#     parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
#     parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
#     parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
#     args = parser.parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

#     # train config
#     config = {}
#     config["softmax_param"] = 1.0
#     config["high"] = 1.0
#     config["num_iterations"] = 12004
#     config["test_interval"] = args.test_interval
#     config["snapshot_interval"] = args.snapshot_interval
#     config["output_for_test"] = True
#     config["output_path"] = "../snapshot/" + args.output_dir
#     if not osp.exists(config["output_path"]):
#         os.mkdir(config["output_path"])
#     config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
#     if not osp.exists(config["output_path"]):
#         os.mkdir(config["output_path"])

#     config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
#     config["loss"] = {"trade_off":1.0, "update_iter":500}
#     if "AlexNet" in args.net:
#         config["network"] = {"name":network.AlexNetFc, \
#             "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
#     elif "ResNet" in args.net:
#         config["network"] = {"name":network.ResNetFc, \
#             "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
#     elif "VGG" in args.net:
#         config["network"] = {"name":network.VGGFc, \
#             "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
#     config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
#                            "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
#                            "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

#     config["dataset"] = args.dset
#     if config["dataset"] == "office":
#         config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
#                           "target":{"list_path":args.t_dset_path, "batch_size":36}, \
#                           "test":{"list_path":args.t_dset_path, "batch_size":4}}
#         if "amazon" in config["data"]["test"]["list_path"]:
#             config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#         else:
#             config["optimizer"]["lr_param"]["init_lr"] = 0.001
#         config["loss"]["update_iter"] = 500
#         config["network"]["params"]["class_num"] = 31
#     elif config["dataset"] == "office-home":
#         config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
#                           "target":{"list_path":args.t_dset_path, "batch_size":36}, \
#                           "test":{"list_path":args.t_dset_path, "batch_size":4}}
#         if "Real_World" in args.s_dset_path and "Art" in args.t_dset_path:
#             config["softmax_param"] = 1.0
#             config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#         elif "Real_World" in args.s_dset_path:
#             config["softmax_param"] = 10.0
#             config["optimizer"]["lr_param"]["init_lr"] = 0.001
#         elif "Art" in args.s_dset_path:
#             config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#             config["high"] = 0.5
#             config["softmax_param"] = 10.0
#             if "Real_World" in args.t_dset_path:
#                 config["high"] = 0.25
#         elif "Product" in args.s_dset_path:
#             config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#             config["high"] = 0.5
#             config["softmax_param"] = 10.0
#             if "Real_World" in args.t_dset_path:
#                 config["high"] = 0.3
#         else:
#             config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#             if "Real_World" in args.t_dset_path:
#                 config["high"] = 0.5
#                 config["softmax_param"] = 10.0
#                 config["loss"]["update_iter"] = 1000
#             else:
#                 config["high"] = 0.5
#                 config["softmax_param"] = 10.0
#                 config["loss"]["update_iter"] = 500
#         config["network"]["params"]["class_num"] = 65
#     elif config["dataset"] == "imagenet":
#         config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
#                           "target":{"list_path":args.t_dset_path, "batch_size":36}, \
#                           "test":{"list_path":args.t_dset_path, "batch_size":4}}
#         config["optimizer"]["lr_param"]["init_lr"] = 0.0003
#         config["loss"]["update_iter"] = 2000
#         config["network"]["params"]["use_bottleneck"] = False
#         config["network"]["params"]["new_cls"] = False
#         config["network"]["params"]["class_num"] = 1000
#     elif config["dataset"] == "caltech":
#         config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
#                           "target":{"list_path":args.t_dset_path, "batch_size":36}, \
#                           "test":{"list_path":args.t_dset_path, "batch_size":4}}
#         config["optimizer"]["lr_param"]["init_lr"] = 0.001
#         config["loss"]["update_iter"] = 500
#         config["network"]["params"]["class_num"] = 256
#     train(config)
    
#     config = {
#         "softmax_param": float,
#         "high": float,
#         "num_iterations": int,
#         "test_interval": int,
#         "loss": {"trade_off": float, "update_iter": int},
#         "new_cls": bool,
#         'optimizer': str,
#         'optimizer_args': dict,
#         'num_classes': int,
#     }