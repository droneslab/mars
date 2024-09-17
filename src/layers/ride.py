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

''' ======================================
        RiDe Layers
    ====================================== '''
class NCA_RI_Add_CrossEntropy(nn.Module): 

    def __init__(self, clsLabels, insLabels, lambda_=0.1, margin=0):
        super().__init__()
        # register a buffer
        self.register_buffer('clsLabels', torch.LongTensor(clsLabels.size(0)))
        self.register_buffer('insLabels', torch.LongTensor(insLabels.size(0)))
        # set the buffer
        self.clsLabels = clsLabels
        self.insLabels = insLabels
        self.margin = margin
        self.lambda_ = lambda_

    def forward(self, x, indexes):

        batchSize = x.size(0)
        # memory size
        n = x.size(1)
        exp = torch.exp(x)

        # cls labels for currect batch
        cls_y = torch.index_select(self.clsLabels, 0, indexes.data).view(batchSize, 1)
        cls_same = cls_y.repeat(1, n).eq_(self.clsLabels)

        # ins labels for current batch
        ins_y = torch.index_select(self.insLabels, 0, indexes.data).view(batchSize, 1)
        ins_same = ins_y.repeat(1, n).eq_(self.insLabels)

        # self prob exclusion, hack with memory for effeciency
        exp.data.scatter_(1, indexes.data.view(-1,1), 0) # P_ii=0
        Z = exp.sum(dim=1)

        p1 = torch.mul(exp, cls_same.float()).sum(dim=1)
        prob1 = torch.div(p1, Z)

        p2 = torch.mul(exp, ins_same.float()).sum(dim=1)
        prob2 = torch.div(p2, Z)

        prob1_masked = torch.masked_select(prob1, prob1.ne(0))
        prob2_masked = torch.masked_select(prob2, prob2.ne(0))

        clsLoss = - prob1_masked.log().sum(0) / batchSize
        insLoss = - self.lambda_ * prob2_masked.log().sum(0) / batchSize

        return clsLoss, insLoss

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out