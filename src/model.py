from __future__ import print_function, division, absolute_import
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_metric_learning import distances, miners, reducers, testers
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import get_ml_loss
from layers.resnext import create_resnext101
from layers.projector import EmbeddingProjector
import numpy as np
from pytorch_grad_cam import EigenCAM
from utils import show_images
import torchvision.transforms.functional as TVF
from losses import AttentionAlignmentRegularization as AAR
import wandb
from utils import untransform


''' Metric Learning Landmark Descriptor Network '''
class MetricLDN(pl.LightningModule):
    def __init__(self, args, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.num_classes = args.datamodule.train_dataset.num_classes*args.datamodule.nper_class
        self.conv_type = args.conv_type
        self.att_type = args.att_type
        self.ml_loss_type = args.ml_loss
        self.aar = args.aar
        self.save_hyperparameters()
        
        # Miner
        self.miner = miners.MultiSimilarityMiner()
        
        # Models
        # self.feature_extractor = create_resnext101(self.conv_type, self.att_type)
        # self.projector = EmbeddingProjector(in_channels=2048)
        
        # import sys
        # sys.path.append('/home/drones/landmark_recognition/src/rebuttal/MixVPR')
        # from main import VPRModel
        # # Note that images must be resized to 320x320
        # self.model = VPRModel(backbone_arch='resnet101', 
        #                 layers_to_crop=[4],
        #                 agg_arch='MixVPR',
        #                 agg_config={'in_channels' : 1024,
        #                             'in_h' : 20,
        #                             'in_w' : 20,
        #                             'out_channels' : 256,
        #                             'mix_depth' : 4,
        #                             'mlp_ratio' : 1,
        #                             'out_rows' : 2},
        #                 )
        
        self.model = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone="ResNet101", fc_output_dim=512)

        
        # Loss functions
        self.ml_loss_func, self.ml_loss_requires_opt = get_ml_loss(self.ml_loss_type, self.num_classes, args.datamodule.train_dataset)

        # Enable attention alignment regularization (proposed method)
        if self.aar:
            self.ch_gamma = args.ch_gamma
            self.sp_gamma = args.sp_gamma
            self.aar_block1 = AAR(256,  ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.aar_block2 = AAR(512,  ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.aar_block3 = AAR(1024, ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.aar_block4 = AAR(2048, ch_gamma=self.ch_gamma, sp_gamma=self.sp_gamma)
            self.aar_str = f'_MARsCosineCh{self.ch_gamma}Sp{self.sp_gamma}'
            # self.aar_str = f'_MARsCosineVariable'
        
        # If using losses that require optimizing parameters 
        if self.ml_loss_requires_opt:
            self.automatic_optimization = False
            
        # Training data logging variables
        self.seen_epoch = -1
        self.class_to_save = 0
        self.saved_training_data = {
            'class_to_save': self.class_to_save,
            'images_a': [], 'images_b': [],
            'T128_a':   [], 'T128_b':   [],
            'T32_a':    [], 'T32_b':    [],
            'T16_a':    [], 'T16_b':    [],
            'T8_a':     [], 'T8_b':     [],
            'T4_a':     [], 'T4_b':     [],
            
            'layer1s_a': [], 'layer1s_b': [],
            'layer2s_a': [], 'layer2s_b': [],
            'layer3s_a': [], 'layer3s_b': [],
            'layer4s_a': [], 'layer4s_b': [],
            
            'cam_layer1s_a': [], 'cam_layer1s_b': [],
            'cam_layer2s_a': [], 'cam_layer2s_b': [],
            'cam_layer3s_a': [], 'cam_layer3s_b': [],
            'cam_layer4s_a': [], 'cam_layer4s_b': [],
        }
        
    def __str__(self):
        model_str = f'{self.conv_type}_{self.att_type}_{self.ml_loss_type}'
        if self.aar:
            model_str += self.aar_str
        return model_str
    
    # Save heatmaps for attention + EigenCAM during training so we can see how they evolve
    def save_training_data(self, xs, ys, T128,T32,T16,T8,T4, features, attentions, idxs):
        idx_a = idxs[0]
        idx_b = idxs[1]
        self.saved_training_data['images_a'].append( xs[idx_a,:].detach().cpu() ) # [C, H, W]
        self.saved_training_data['images_b'].append( xs[idx_b,:].detach().cpu() )
        self.saved_training_data['T128_a'].append(T128[idx_a])
        self.saved_training_data['T128_b'].append(T128[idx_b])
        self.saved_training_data['T32_a'].append(T32[idx_a])
        self.saved_training_data['T32_b'].append(T32[idx_b])
        self.saved_training_data['T16_a'].append(T16[idx_a])
        self.saved_training_data['T16_b'].append(T16[idx_b])
        self.saved_training_data['T8_a'].append(T8[idx_a])
        self.saved_training_data['T8_b'].append(T8[idx_b])
        self.saved_training_data['T4_a'].append(T4[idx_a])
        self.saved_training_data['T4_b'].append(T4[idx_b])
    
        layer1_a = attentions[0][idx_a,-1,...].detach().cpu() # [C, H, W]
        layer1_b = attentions[0][idx_b,-1,...].detach().cpu()
        layer2_a = attentions[1][idx_a,-1,...].detach().cpu()
        layer2_b = attentions[1][idx_b,-1,...].detach().cpu()
        layer3_a = attentions[2][idx_a,-1,...].detach().cpu()
        layer3_b = attentions[2][idx_b,-1,...].detach().cpu()
        layer4_a = attentions[3][idx_a,-1,...].detach().cpu()
        layer4_b = attentions[3][idx_b,-1,...].detach().cpu()
        self.saved_training_data['layer1s_a'].append(layer1_a)
        self.saved_training_data['layer1s_b'].append(layer1_b)
        self.saved_training_data['layer2s_a'].append(layer2_a)
        self.saved_training_data['layer2s_b'].append(layer2_b)
        self.saved_training_data['layer3s_a'].append(layer3_a)
        self.saved_training_data['layer3s_b'].append(layer3_b)
        self.saved_training_data['layer4s_a'].append(layer4_a)
        self.saved_training_data['layer4s_b'].append(layer4_b)
        
        images = xs[idxs,:]
        cams1 = EigenCAM(self.feature_extractor, target_layers=[self.feature_extractor.layer1[-1]], use_cuda=False)(input_tensor=images, aug_smooth=True, eigen_smooth=True)
        cams2 = EigenCAM(self.feature_extractor, target_layers=[self.feature_extractor.layer2[-1]], use_cuda=False)(input_tensor=images, aug_smooth=True, eigen_smooth=True)
        cams3 = EigenCAM(self.feature_extractor, target_layers=[self.feature_extractor.layer3[-1]], use_cuda=False)(input_tensor=images, aug_smooth=True, eigen_smooth=True)
        cams4 = EigenCAM(self.feature_extractor, target_layers=[self.feature_extractor.layer4[-1]], use_cuda=False)(input_tensor=images, aug_smooth=True, eigen_smooth=True)
        
        self.saved_training_data['cam_layer1s_a'].append(torch.Tensor(cams1[0,:]).unsqueeze(0).detach().cpu()) # [C, H, W]
        self.saved_training_data['cam_layer2s_a'].append(torch.Tensor(cams2[0,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer3s_a'].append(torch.Tensor(cams3[0,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer4s_a'].append(torch.Tensor(cams4[0,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer1s_b'].append(torch.Tensor(cams1[1,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer2s_b'].append(torch.Tensor(cams2[1,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer3s_b'].append(torch.Tensor(cams3[1,:]).unsqueeze(0).detach().cpu())
        self.saved_training_data['cam_layer4s_b'].append(torch.Tensor(cams4[1,:]).unsqueeze(0).detach().cpu())
        
    
    def configure_optimizers(self):
        if self.ml_loss_requires_opt:
            opt_model = optim.Adam(self.parameters(), lr=self.lr)
            opt_loss = optim.Adam(self.ml_loss_func.parameters(), lr=1e-4)
            return [opt_model, opt_loss]
        else:
            return optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        # features = self.feature_extractor(x)
        # embeddings = self.projector(features)
        # return embeddings # Return embeddings that lie on optimized manifold space
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if not self.automatic_optimization:
            opt_model, opt_loss = self.optimizers()
            opt_model.zero_grad()
            opt_loss.zero_grad()
        
        xs,ys,T128,T32,T16,T8,T4 = batch # ys ==> crater number
                        
        # features = self.feature_extractor(xs)
        # features, attentions = self.feature_extractor.features_and_attentions(xs)        
        # embeddings = self.projector(features)
        
        embeddings = self.forward(xs)
                
        # Save training data ONCE per epoch
        # idxs = (ys == self.class_to_save).nonzero(as_tuple=True)[0] # will be two of them
        # if idxs.shape[0] != 0 and self.current_epoch != self.seen_epoch:
        #     self.seen_epoch = self.current_epoch
        #     self.save_training_data(xs,ys,T128,T32,T16,T8,T4,features,attentions,idxs)
        
        ''' Attention Alignment Loss '''
        aar_loss = 0
        # if self.aar:
        #     L_block1 = self.aar_block1(attentions[0], ys, T32)
        #     L_block2 = self.aar_block2(attentions[1], ys, T16)
        #     L_block3 = self.aar_block3(attentions[2], ys, T8)
        #     L_block4 = self.aar_block4(attentions[3], ys, T4)
        #     aar_loss += L_block1 + L_block2 + L_block3 + L_block4
            
        #     self.log('L_AAR_block1', L_block1, prog_bar=False)
        #     self.log('L_AAR_block2', L_block2, prog_bar=False)
        #     self.log('L_AAR_block3', L_block3, prog_bar=False)
        #     self.log('L_AAR_block4', L_block4, prog_bar=False)
        #     self.log('AAR', aar_loss, prog_bar=True)
        
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
    
        total_loss = ml_loss + aar_loss
                
        if not self.automatic_optimization:
            self.manual_backward(total_loss)
            opt_model.step()
            opt_loss.step()
        return total_loss
    