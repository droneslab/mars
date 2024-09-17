import torch
from torch.utils.data.sampler import Sampler
import math
import numpy as np

# Modified from:
# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
class CustomMPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    # NOTE: Assuming only 1 unique label in labels list
    def __init__(self, labels, m, batch_size, shuffle=True):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        self.indices = np.arange(len(labels))
        self.shuffle = shuffle

    def __len__(self):
        num = len(self.indices)*self.m_per_class
        to_fill = num%self.batch_size
        return num + to_fill

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        idx_list = np.repeat(self.indices, self.m_per_class)
        num_to_fill = len(idx_list)%self.batch_size
        idx_list = np.append(idx_list, idx_list[:num_to_fill])
        return iter(idx_list)
