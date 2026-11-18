import torch
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d,Conv1d
from torch.nn import ConvTranspose2d,ConvTranspose1d
from .complexLayers import *
import torch.nn as nn

class CVnet(nn.Module):
    """
    完整CV-Res-ADMM-net实现

    """
    def __init__(self, in_chans, out_chans, chans=64, num_pool_layers=4, drop_prob=0.0):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.conv1 = ComplexConv2d(in_chans,out_chans)
        self.conv2 = ComplexConv2d(out_chans,out_chans)
        self.relu = ComplexReLU()
        self.conv3 = ComplexConv2d(out_chans,out_chans)
        self.conv4 = ComplexConv2d(out_chans,in_chans)


    def forward(self, image):
        p1 = self.conv1(image)
        p2 = p1 + self.conv3(self.relu(self.conv2(p1)))
        return image + self.conv4(p2)