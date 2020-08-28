import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
# import torch.nn._functions as tnnf

# NOTE: the code in here is for using binarized neural networks and is copied from here: 
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py 


def Quantize(tensor, quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det': # deterministic binarization via sign function
        return tensor.sign() 
    else: # stochastic binarization via hard sigmoid
        if 'cuda' in str(tensor): 
            return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).cuda().add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
        else: 
            return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        if 'stochastic_bin' in kwargs.keys(): 
            if kwargs['stochastic_bin']: 
                self.quant_mode = 'stoch'
            else: 
                self.quant_mode = 'det'
            del kwargs['stochastic_bin']
        else: 
            self.quant_mode = 'det' 

        if 'num_features' in kwargs.keys(): 
            self.num_features = kwargs['num_features']
            del kwargs['num_features']
        else: 
            self.num_features = 784 # as it was originally for mnist
            
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        # self.num_features = 9

    def forward(self, input):

        if input.size(1) != self.num_features: 
            input.data=Binarize(input.data, quant_mode=self.quant_mode)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        self.weight.data=Binarize(self.weight.org, quant_mode=self.quant_mode)

        out = nn.functional.linear(input, self.weight)

        if not (self.bias is None):
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
