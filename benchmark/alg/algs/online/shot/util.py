import torch
from torch import nn 
from scipy.spatial.distance import cdist
import numpy as np
from torch.utils.data import TensorDataset


def obtain_label(not_shuffled_dataloader, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()
    
    all_x = None
    
    start_test = True
    with torch.no_grad():
        # print(not_shuffled_dataloader)
        iter_test = iter(not_shuffled_dataloader)
        # print(iter_test)
        for _ in range(len(not_shuffled_dataloader)):
            data = next(iter_test)
            inputs = data[0]
            # labels = data[1]
            inputs = inputs.cuda()
            feas = feature_extractor(inputs)
            outputs = classifier(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                # all_label = labels.float()
                all_x = inputs.cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # all_label = torch.cat((all_label, labels.float()), 0)
                all_x = torch.cat((all_x, inputs.cpu()), 0)

    # all_output = nn.Softmax(dim=1)(all_output)
    # _, predict = torch.max(all_output, 1)
    # # first_acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    # all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    # all_fea = all_fea.float().cpu().numpy()

    # K = all_output.size(1)
    # aff = all_output.float().cpu().numpy()
    # initc = aff.transpose().dot(all_fea)
    # initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    # dd = cdist(all_fea, initc, 'cosine')
    # pred_label = dd.argmin(axis=1)
    # # last_acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    # for round in range(1):
    #     aff = np.eye(K)[pred_label]
    #     initc = aff.transpose().dot(all_fea)
    #     initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    #     dd = cdist(all_fea, initc, 'cosine')
    #     pred_label = dd.argmin(axis=1)
    #     # last_acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    
    
    all_output = nn.Softmax(dim=1)(all_output)
    # ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    # unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    distance = 'cosine'
    if True:
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    
    threshold = 0.
    labelset = np.where(cls_count>threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    # print(pred_label)
        
    res_x, res_y = all_x, torch.LongTensor(pred_label)
    new_dataset = TensorDataset(res_x, res_y)

    return new_dataset
