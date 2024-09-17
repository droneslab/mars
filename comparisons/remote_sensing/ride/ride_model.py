import torch
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

from ride_loss import LinearAverage
from ride_loss import NCA_RI_Add_CrossEntropy
from model import ResNet18, ResNet50, BNInception, ResNet34
from modelSCCov import SCCov_Res34_emb, SCCov_Res50_emb


class RiDe(pl.LightningModule):
    def __init__(self, dataset, model_str='resnet34'):
        super().__init__()
        if model_str == 'resnet50':
            self.model = ResNet50(dim=512)
        elif model_str == 'resnet34':
            self.model = ResNet34(dim=512)
        elif model_str == 'sccov_res34':
            self.model = SCCov_Res34_emb(dim=512)
        elif model_str == 'sccov_res50':
            self.model = SCCov_Res50_emb(dim=512)
        else:
            self.model = BNInception(dim=512)
            
        self.model = self.model.cuda()
        
        ndata = len(dataset.train_dataset)
        self.lemniscate = LinearAverage(512, ndata, 0.05, 0.5).cuda()
        
        cls_y_true = np.asarray(dataset.train_dataset.labels)
        ins_y_true = np.arange(len(cls_y_true))
        
        self.criterion = NCA_RI_Add_CrossEntropy(torch.LongTensor(cls_y_true), torch.LongTensor(ins_y_true), 0.1, 0.0 / 0.05).cuda()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        xs,ys = batch
        feature = self.model(xs)
        output = self.lemniscate(feature, ys)
        clsLoss, insLoss = self.criterion(output, ys)
        loss = clsLoss + insLoss
        
        self.log('RiDeLoss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), 1e-4, momentum=0.9, weight_decay=1e-4, nesterov=True)