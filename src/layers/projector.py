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


def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(pl.LightningModule):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class EmbeddingProjector(pl.LightningModule):
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.gem_pool = GeM()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.batchnorm = torch.nn.BatchNorm1d(out_channels)
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        x = self.gem_pool(x)
        x = x.reshape((x.shape[0], x.shape[1]))
        x = self.linear(x)
        x = self.batchnorm(x)
        x = self.prelu(x)
        return x