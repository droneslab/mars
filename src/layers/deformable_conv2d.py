from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Function
import torchvision
from layers.attention.cbam import CBAM
from layers.attention.se import SEModule

# DeformableConv2d class: https://github.com/yjh0410/PyTorch_DCNv2/blob/main/DCN/dcnv2.py
class DeformableConv2d(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(DeformableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.use_bias = bias
        self.dilation = dilation
        self.groups = groups
        
        self.conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, 
                                                 padding, dilation, groups, bias)

        # offset conv
        self.conv_offset_mask = nn.Conv2d(in_channels, 
                                          3 * kernel_size * kernel_size,
                                          kernel_size=kernel_size, 
                                          stride=stride,
                                          padding=self.padding, 
                                          bias=self.use_bias)
        
        # init        
        self._init_weight()

    def _init_weight(self):
        # init offset_mask conv
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        if self.use_bias:
            nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.conv(x, offset, None) # None for v1, mask above for v2