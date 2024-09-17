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
class RICConv2d(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(RICConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.dilation = dilation
        self.groups = groups
        
        self.conv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, 
                                                 padding, dilation, groups, bias)
            
    def generate_coordinates(self, batch_size, input_height, input_width, kernel_size):
        
        n = (kernel_size-1)/2
        
        coords = torch.zeros(input_height, input_width, 2*kernel_size*kernel_size)
        
        parameters = torch.zeros(3)
        parameters[0] = batch_size
        parameters[1] = input_height
        parameters[2] = input_width
        
        center = torch.zeros(2)
        center[0]=torch.sub(torch.div(parameters[1], 2.0), 0.5)
        center[1]=torch.sub(torch.div(parameters[2], 2.0), 0.5)
            
        x_grid = torch.arange(0, parameters[1])
        y_grid = torch.arange(0, parameters[2]) 
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
        
        #coords[:,:,8]=grid_x
        #coords[:,:,9]=grid_y
               
        delta_x = torch.sub(grid_x, center[0])
        delta_y = torch.sub(grid_y, center[1])
        PI = torch.mul(torch.Tensor([math.pi]), 2.0)
        theta=torch.atan2(delta_y, delta_x) % PI[0]
        theta=torch.round(10000.*theta)/10000.
        
        # THIS IS A 3x3 grid ({-1,0,1} x {-1,0,1}), handles all coords except 0,0
        
        # 1,1
        coords[:,:,0]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0),1.0) # X
        coords[:,:,1]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0),1.0) # Y   
        
        # 1,0
        coords[:,:,2]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0),1.0)
        coords[:,:,3]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0),0.0)
        
        # 1,-1
        coords[:,:,4]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0),1.0)
        coords[:,:,5]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0),-1.0)
        
        # 0,1
        coords[:,:,6]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0),0.0)
        coords[:,:,7]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0),1.0)
        
        # NO CENTER POINT (0,0)
        
        # 0,-1
        coords[:,:,10]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),1.0),0.0)
        coords[:,:,11]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),1.0),-1.0)
        
        # -1,1
        coords[:,:,12]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0),-1.0)
        coords[:,:,13]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0),1.0)
        
        # -1,0
        coords[:,:,14]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),1.0),-1.0)
        coords[:,:,15]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),1.0),0.0)
        
        # -1,-1
        coords[:,:,16]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),1.0),-1.0)
        coords[:,:,17]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),1.0),-1.0)
        
        coords=coords.expand(batch_size,-1,-1,-1) 
        coords=coords.permute(0, 3, 1, 2)
        return coords

    def forward(self, x):
        # Calculate output conv shape and create RIC-C grid as deform offset
        H_out = math.floor(((x.shape[2] + 2 * self.padding - 1 * (self.kernel_size-1)-1 ) / (self.stride)) + 1 )
        W_out = math.floor(((x.shape[3] + 2 * self.padding - 1 * (self.kernel_size-1)-1 ) / (self.stride)) + 1 )
        coords = self.generate_coordinates(x.shape[0], H_out, W_out, self.kernel_size).cuda()
        return self.conv(x, coords, None)
    