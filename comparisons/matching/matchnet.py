import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from model import ResNext

# class MatchNet(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.feature_net = ResNext('se', 'none', 'none', None)
#         self.metric_net = MetricNet()

#     def forward(self, x):
#         return self.feature_net(x)

#     def training_step(self, batch, batch_idx, optimizer_idx=None):
#         xs,ys,trans = batch
#         embeddings = self.feature_net(xs)
#         # Form positive, negative pairs
#         pos_idxs = torch.arange(0,ys.shape[0], dtype=torch.long).reshape((ys.shape[0]//2,2))
#         neg_idxs = torch.roll(pos_idxs, 1)
#         # Output from the two towers are concatenated as the metric network's input
#         pos_embs = embeddings[pos_idxs].reshape((pos_idxs.shape[0], 1024))
#         neg_embs = embeddings[neg_idxs].reshape((neg_idxs.shape[0], 1024))
#         all_embs = torch.cat((pos_embs, neg_embs), 0)
#         # Feed into metric net and calculate loss. First 16 pairs positive, rest negative
#         match_scores = self.metric_net(all_embs)
#         # Eq. 2
#         y_pred = torch.exp(match_scores[:,1]) / (torch.exp(match_scores[:,0]) + torch.exp(match_scores[:,1]))
#         t = torch.ones((ys.shape[0]//2))
#         n = torch.zeros((ys.shape[0]//2))
#         y_true = torch.cat((t,n),0).cuda()
#         # Eq. 1
#         loss = -torch.sum( y_true * torch.log(y_pred) + (1-y_true) * torch.log(1-y_pred) ) / ys.shape[0]
#         self.log('Cross Entropy Loss', loss, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return optim.SGD(self.parameters(), lr=0.01)

class MetricNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024,1024)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(1024,1024)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(1024,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class MetricNetLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.metric_net = MetricNet()
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, embeddings, ys, indices_tuple):
        a1, p, a2, n = indices_tuple
        # Form positive, negative pairs
        pos_embs1 = embeddings[a1,:]
        pos_embs2 = embeddings[p, :]
        neg_embs1 = embeddings[a2,:]
        neg_embs2 = embeddings[n, :]
        # Output from the two towers are concatenated as the metric network's input
        pos_embs = torch.cat((pos_embs1,pos_embs2), dim=1)
        neg_embs = torch.cat((neg_embs1,neg_embs2), dim=1)
        all_embs = torch.cat((pos_embs, neg_embs), 0)
        # Feed into metric net and calculate loss. First 16 pairs positive, rest negative
        match_scores = self.metric_net(all_embs)
        # Eq. 2
        y_pred = torch.exp(match_scores[:,1]) / (torch.exp(match_scores[:,0]) + torch.exp(match_scores[:,1]))
        t = torch.ones((a1.shape[0]))
        n = torch.zeros((a2.shape[0]))
        y_true = torch.cat((t,n),0).cuda()
        # Eq. 1
        loss = -torch.sum( y_true * torch.log(y_pred) + (1-y_true) * torch.log(1-y_pred) ) / (a1.shape[0] + a2.shape[0])
        return loss
    