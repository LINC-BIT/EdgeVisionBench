import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torchvision import models

import config_supervised 

Es_dims = config_supervised.settings['Es_dims']
cnn_to_use = config_supervised.settings['cnn_to_use']

class CustomResNet(nn.Module):

    def __init__(self):

        super(CustomResNet, self).__init__()

        temp_resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]]) # Upto the avgpool layer

    def forward(self, x):

        feats = self.features(x)
        return feats.view((x.shape[0], 2048))


class modnet(nn.Module):
    
    def __init__(self, num_C, num_Ct_uk, cnn=cnn_to_use, additional_components=[]): 
        
        super(modnet, self).__init__()

        # Frozen initial conv layers
        if cnn=='resnet50':
            self.M = CustomResNet()
        else:
            raise NotImplementedError('Not implemented for ' + str(cnn))

        self.Es = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,Es_dims),
            nn.ELU(),
            nn.Linear(Es_dims, Es_dims),
            nn.BatchNorm1d(Es_dims),
            nn.ELU(),
        )

        self.Et = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ELU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024,Es_dims),
            nn.ELU(),
            nn.Linear(Es_dims, Es_dims),
            nn.BatchNorm1d(Es_dims),
            nn.ELU(),
        )

        self.Gn = nn.Sequential(
            nn.Linear(Es_dims, 45)
        )

        self.Gs = nn.Sequential(
            nn.Linear(Es_dims, num_C)
        )
        

        self.components = {
            'M': self.M,
            'Es': self.Es,
            'Et': self.Et,
            'Gs': self.Gs,
            'Gn': self.Gn,
        }

    def forward(self, x, which_fext='original'):
        raise NotImplementedError('Implemented a custom forward in train loop')

