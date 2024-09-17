from __future__ import print_function, division, absolute_import
import torch.nn as nn
import pytorch_lightning as pl
import torch

''' ======================================
        Pixel-wise Top-k Attention Pooling
    ====================================== '''
# From the paper: Contrastive Attention Maps for Self-supervised Co-localization
class ChannelAttention(pl.LightningModule):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class PTAP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.k = int(2048*0.7)
        # self.k = int(2048)
        self.ca = ChannelAttention(2048)
        
    def forward(self, x):
        Fw = self.ca(x)
        Fw,_ = torch.sort(Fw, dim=1, descending=True) # Sort activations by priority
        return (1/self.k)*torch.sum(Fw[:, :self.k, :, :], dim=1)