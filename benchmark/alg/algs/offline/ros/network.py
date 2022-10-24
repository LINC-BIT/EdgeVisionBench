from torch import nn 
import torch


class Discriminator(nn.Module):
    def __init__(self, in_features, num_classes=None, num_ssl_classes_for_each_class=None):
        super(Discriminator, self).__init__()
        self.n = num_classes
        def f():
            return nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(256, num_ssl_classes_for_each_class))                   

        def f_feat():
            return nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2, inplace=True))                   
               
        for i in range(num_classes):
            self.__setattr__('discriminator_%04d'%i, f())
            self.__setattr__('discriminator_feat_%04d'%i, f_feat())
    
    def forward(self, x):
        outs = [self.__getattr__('discriminator_%04d'%i)(x) for i in range(self.n)]
        outs_feat = [self.__getattr__('discriminator_feat_%04d'%i)(x) for i in range(self.n)]

        return torch.cat(outs, dim=-1),torch.cat(outs_feat, dim=-1)
    