import torch
import copy
import numpy as np
from torch import nn

class GradientReverseLayer(torch.autograd.Function):
    def __init__(self):
        self.coeff = 1.0
    def forward(self, input):
        return input
    def backward(self, gradOutput):
        return -self.coeff * gradOutput

class GradientReverseModule(nn.Module):
    def __init__(self):
        super(GradientReverseModule, self).__init__()
        # self.scheduler = scheduler
        self.global_step = 0.0
        self.grl = GradientReverseLayer()
        
    def scheduler(self, step):
        def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
            ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
            return float(ans)
        return aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000)
        
    def forward(self, x):
        coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        self.grl.coeff = coeff
        return self.grl.forward(x)