from __future__ import print_function, division, absolute_import
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_metric_learning import miners
from losses import get_ml_loss
from layers.resnext import create_resnext101
from layers.projector import EmbeddingProjector
from losses import MARs


''' Metric Learning Landmark Descriptor Network '''
class MetricLDN(pl.LightningModule):
    def __init__(self, cfg, datamodule, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.num_classes = datamodule.train_dataset.num_classes*datamodule.nper_class
        self.ml_loss_type = cfg['metric_learning_loss']
        self.mars = cfg['mars_enabled']
        self.save_hyperparameters()
        
        # Miner
        self.miner = miners.MultiSimilarityMiner()
        
        # Models
        self.feature_extractor = create_resnext101(cfg['conv_type'], cfg['att_type'])
        self.projector = EmbeddingProjector(in_channels=2048)
        
        # Loss functions
        self.ml_loss_func, self.ml_loss_requires_opt = get_ml_loss(self.ml_loss_type, self.num_classes, datamodule.train_dataset)

        # Multi-view Attention Regularizations (MARs)
        if self.mars:
            self.ch_gamma = cfg['channel_gamma']
            self.sp_gamma = cfg['spatial_gamma']
            self.mars_block1 = MARs(256,  ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.mars_block2 = MARs(512,  ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.mars_block3 = MARs(1024, ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.mars_block4 = MARs(2048, ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)

        # If using losses that require optimizing parameters 
        if self.ml_loss_requires_opt:
            self.automatic_optimization = False
    
    def configure_optimizers(self):
        if self.ml_loss_requires_opt:
            opt_model = optim.Adam(self.parameters(), lr=self.lr)
            opt_loss = optim.Adam(self.ml_loss_func.parameters(), lr=1e-4)
            return [opt_model, opt_loss]
        else:
            return optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        features = self.feature_extractor(x)
        embeddings = self.projector(features)
        return embeddings # Return embeddings that lie on optimized manifold space

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            opt_model, opt_loss = self.optimizers()
            opt_model.zero_grad()
            opt_loss.zero_grad()
        
        # Batch and inference
        xs,ys,T128,T32,T16,T8,T4 = batch # ys ==> landmark index number                        
        features, attentions = self.feature_extractor.features_and_attentions(xs)        
        embeddings = self.projector(features)        
        
        ''' Attention Alignment Loss '''
        mars_loss = 0
        if self.mars:
            L_block1 = self.mars_block1(attentions[0], ys, T32)
            L_block2 = self.mars_block2(attentions[1], ys, T16)
            L_block3 = self.mars_block3(attentions[2], ys, T8)
            L_block4 = self.mars_block4(attentions[3], ys, T4)
            mars_loss += L_block1 + L_block2 + L_block3 + L_block4
        
        ''' Metric Learning Loss '''
        if self.ml_loss_type in ['proxynca++', 'synproxy', 'pnp']:
            ml_loss = self.ml_loss_func(embeddings, ys)
        else:
            # Do sample mining for hard-negatives 
            #   ap,p should always be the two class-instances in the batch given by the sampler
            ap,p,an,n = self.miner(embeddings, ys) # Miner gives (anchor-positives, postives, anchor-negatives, negatives)
            ml_loss = self.ml_loss_func(embeddings, ys, (ap,p,an,n))
            # Skip any bad updates
            if torch.isnan(ml_loss).any():
                return None
            
        self.log(f'{self.ml_loss_type}', ml_loss, prog_bar=True)
    
        total_loss = ml_loss + mars_loss
                
        if not self.automatic_optimization:
            self.manual_backward(total_loss)
            opt_model.step()
            opt_loss.step()
        return total_loss
    