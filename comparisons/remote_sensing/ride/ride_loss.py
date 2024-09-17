import pytorch_lightning as pl
import numpy as np
import torch
from layers.ride import LinearAverage, NCA_RI_Add_CrossEntropy

class RiDeLoss(pl.LightningModule):
    def __init__(self, train_dataset):
        super().__init__()
        ndata = len(train_dataset)
        self.lemniscate = LinearAverage(512, ndata, 0.05, 0.5).cuda()
        cls_y_true = np.asarray(train_dataset.labels*2)
        ins_y_true = np.asarray(train_dataset.labels*2)
        self.criterion = NCA_RI_Add_CrossEntropy(torch.LongTensor(cls_y_true), torch.LongTensor(ins_y_true), 0.1, 0.0 / 0.05).cuda()

    def forward(self, features, ys):
        features = torch.nn.functional.normalize(features)
        outputs = self.lemniscate(features, ys)
        clsLoss, insLoss = self.criterion(outputs, ys)
        loss = clsLoss + insLoss
        return loss
