from __future__ import print_function, division, absolute_import
import torch.nn as nn
import pytorch_lightning as pl

''' ======================================
        Squeeze-Excitation Module
    ====================================== '''
class SEModule(pl.LightningModule):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.weights = None
        self.output = None
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        out = module_input * x
        self.weights = x # Save output to grab it later
        self.output = out # Save output to grab it later
        return out