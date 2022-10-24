import torch
import numpy as np


class AdversarialLayer(torch.autograd.Function):
  def __init__(self, high_value=1.0):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = high_value
    self.max_iter = 10000.0
    
  def forward(self, input):
    self.iter_num += 1
    output = input * 1.0
    return output

  def backward(self, gradOutput):
    self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
    return -self.coeff * gradOutput