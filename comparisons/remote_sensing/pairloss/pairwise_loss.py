import torch
import pytorch_lightning as pl

class PairLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tow = 0.05 # default triplet margin loss margin (pytorch ML)
    
    def forward(self, features, labels):
        bSize = features.shape[0]
        combos = torch.combinations(torch.arange(bSize), r=2)
        loss = 0
        for i,j in combos:
            xi = features[i]
            xj = features[j]
            yi = labels[i]
            yj = labels[j]
            yij = 1 if yi == yj else -1
            norm = torch.linalg.vector_norm((xi-xj))
            l = 0.05 - yij * (self.tow - norm)
            loss += torch.max(torch.tensor(0),l)
        return loss
