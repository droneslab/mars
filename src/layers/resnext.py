"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

SEResNeXt101 model from
https://www.kaggle.com/code/satian/seresnext101-pytorch-starter/notebook
"""
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
from .attention.cbam import CBAM
from .attention.se import SEModule
from .attention.ca import CoordAtt
from .deformable_conv2d import DeformableConv2d
from .ric_conv2d import RICConv2d
from .bottlenecks import Bottleneck

CONV = None
ATT = None

# Bottleneck inheritence is only for forward method
class ResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4
    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4,
                 conv_type=nn.Conv2d, att_type=SEModule):
        super(ResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv_type(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.att = att_type(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

class ResNeXt101(pl.LightningModule):

    def __init__(self, bottleneck, conv_type, att_type, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1):
        """
        Parameters
        ----------
        bottleneck (pl.LightningModule): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        """
        super(ResNeXt101, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', conv_type(3, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', conv_type(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', conv_type(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            bottleneck,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
            conv_type=conv_type,
            att_type=att_type
        )
        self.layer2 = self._make_layer(
            bottleneck,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            conv_type=conv_type,
            att_type=att_type
        )
        self.layer3 = self._make_layer(
            bottleneck,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            conv_type=conv_type,
            att_type=att_type
        )
        self.layer4 = self._make_layer(
            bottleneck,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
            conv_type=conv_type,
            att_type=att_type
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0, conv_type=nn.Conv2d, att_type=SEModule):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample, conv_type=conv_type, att_type=att_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction, conv_type=conv_type, att_type=att_type))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def features_and_attentions(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)   
        x = self.layer4(x)
        
        if self.att_type == 'se' or self.att_type == 'cbam':
            aw1 = torch.stack([x.att.weights for x in self.layer1], dim=1)
            aw2 = torch.stack([x.att.weights for x in self.layer2], dim=1)
            aw3 = torch.stack([x.att.weights for x in self.layer3], dim=1)
            aw4 = torch.stack([x.att.weights for x in self.layer4], dim=1)
        
        if self.att_type == 'ca':
            aw1_h = torch.stack([x.att.h_weights for x in self.layer1], dim=1)
            aw1_w = torch.stack([x.att.w_weights for x in self.layer1], dim=1)
            aw1 = aw1_h*aw1_w # batch, num resnext blocks, C, H, W
            aw2_h = torch.stack([x.att.h_weights for x in self.layer2], dim=1)
            aw2_w = torch.stack([x.att.w_weights for x in self.layer2], dim=1)
            aw2 = aw2_h*aw2_w # batch, num resnext blocks, C, H, W
            aw3_h = torch.stack([x.att.h_weights for x in self.layer3], dim=1)
            aw3_w = torch.stack([x.att.w_weights for x in self.layer3], dim=1)
            aw3 = aw3_h*aw3_w # batch, num resnext blocks, C, H, W
            aw4_h = torch.stack([x.att.h_weights for x in self.layer4], dim=1)
            aw4_w = torch.stack([x.att.w_weights for x in self.layer4], dim=1)
            aw4 = aw4_h*aw4_w # batch, num resnext blocks, C, H, W

        return x, [aw1,aw2,aw3,aw4]
    
    def get_all_features(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x1,x2,x3,x4,x5]

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.features(x)
        return x

''' ======================================
        Model Instantiation
    ====================================== '''
def create_resnext101(conv_type, att_type):
    conv_types = {
        'conv': nn.Conv2d,
        # 'deform': torchvision.ops.DeformConv2d,
        'deform': DeformableConv2d,
        'ric': RICConv2d
    }
    att_types = {
        'se': SEModule,
        'cbam': CBAM,
        'ca': CoordAtt
    }
    ctype = conv_types[conv_type]
    atype = att_types[att_type]
    model = ResNeXt101(ResNeXtBottleneck, ctype, atype, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0)
    model.conv_type = conv_type
    model.att_type = att_type
    return model
