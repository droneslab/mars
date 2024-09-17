import torch
from torch.nn import init
import numpy as np
import math
from pytorch_metric_learning.losses import BaseMetricLossFunction, GenericPairLoss, ContrastiveLoss, TripletMarginLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
import pytorch_metric_learning.losses as pml_losses
import pytorch_lightning as pl
import torchvision as tv
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from utils import show_images, untransform, show_image
from os.path import abspath
# import sys
# sys.path.append(abspath('../comparisons/matching/'))
# sys.path.append(abspath('../comparisons/remote_sensing/pairloss/'))
# sys.path.append(abspath('../comparisons/remote_sensing/ride/'))
# from matchnet import MetricNetLoss
# from pairwise_loss import PairLoss
# from ride_loss import RiDeLoss
from layers.projector import gem, GeM
from layers.attention.cbam import ChannelPool
from layers.projector import EmbeddingProjector
from layers.pyramid_pool import SpatialPyramidPooling
import torchvision.transforms.functional as TFV


class AttentionAlignmentRegularization(pl.LightningModule):
    def __init__(self, in_channels, ch_gamma, sp_gamma):
        super().__init__()
        self.c_gam = ch_gamma
        self.s_gam = sp_gamma
        
        # Reduction 
        self.reduc = nn.Conv2d(in_channels, in_channels//4, 1)
        
        # Channel alignment
        self.gem_c = GeM()
        self.bn_c = nn.BatchNorm1d(in_channels//4)
        self.prelu_c = nn.PReLU()
        
        # Spatial alignment (X)
        self.gem_w = GeM()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.bn_w = nn.BatchNorm1d(in_channels//4)
        self.prelu_w = nn.PReLU()
        
        # Spatial alignment (Y)
        self.gem_h = GeM()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.bn_h = nn.BatchNorm1d(in_channels//4)
        self.prelu_h = nn.PReLU()
                
    def forward(self, Xs, ys, T_normals): # Xs shape => [B, N_res, C, H, W]
        idxA = torch.arange(0,ys.shape[0],2)
        idxB = torch.arange(1,ys.shape[0],2)
        
        B,N,C,H,W = Xs.shape
        x = untransform(Xs, T_normals).reshape((B*N, C, H, W))
        y = ys.repeat(N)
        idxA = idxA.repeat(N)
        idxB = idxB.repeat(N)
        
        x = self.reduc(x)
    
        # Channel vector
        c = self.gem_c(x).squeeze()
        c = self.prelu_c(self.bn_c(c))
        ch_loss = self.c_gam * (1-torch.mean(F.cosine_similarity(c[idxA,:], c[idxB,:])))
        
        # Spatial width (X) vector
        w = self.gem_w(self.pool_w(x)).squeeze()
        w = self.prelu_w(self.bn_w(w))
        w_loss = 1-torch.mean(F.cosine_similarity(w[idxA,:], w[idxB,:]))
        
        # Spatial height (Y) vector
        h = self.gem_h(self.pool_h(x)).squeeze()
        h = self.prelu_h(self.bn_h(h))
        h_loss = 1-torch.mean(F.cosine_similarity(h[idxA,:], h[idxB,:]))
        
        sp_loss = self.s_gam * (h_loss + w_loss)
                                
        return ch_loss + sp_loss

        
'''
Direction Regularized (DR) loss from the paper:
    Moving in the Right Direction: A Regularization for Deep Metric Learning
     - Deen Dayal Mohan et. al.
''' 
class DRMultiSimilarity(GenericPairLoss):
    def __init__(self):
        super().__init__(mat_based_loss=True)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # Per paper results, table 1 settings
        self.alpha = 2
        self.beta = 50
        self.lam = 0.7
        self.gamma = 0.3
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "lam", "gamma"], is_stat=False
        )
        
        # Keep reference of positive pairs for training
        # NOTE: This only works with batches that yield 2 instances per class with congruency (e.g. [0,0, 1,1, 2,2, 3,3, ...])
        self.pos_idxs = None
        
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        if self.pos_idxs is None:
            bsize = embeddings.shape[0]
            pidxs1 = torch.arange(0,bsize, dtype=torch.long).reshape((bsize//2,2))
            pidxs2 = torch.flip(pidxs1, dims=[1])
            self.pos_idxs = torch.stack((pidxs1, pidxs2), dim=1).reshape((bsize,2)).cuda()
            
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = indices_tuple
        pn = self.pos_idxs[a2,:][:,1]
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        pos_exp = self.distance.margin(mat, self.lam)
        neg_exp = self.distance.margin(self.lam, mat)
        
        a_embs = F.normalize(embeddings[a2])
        n_embs = F.normalize(embeddings[n])
        pn_embs = F.normalize(embeddings[pn])
        cos = self.gamma * torch.mean(((n_embs-a_embs) @ (pn_embs-a_embs).t()), dim=1)
        neg_exp[neg_mask.bool()] -= cos
        
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        
        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }
    
    def get_default_distance(self):
        return CosineSimilarity()

class ProxyNCA_pp(pl.LightningModule):
    def __init__(self, num_classes, embedding_size, scale=3, **kwargs):
        nn.Module.__init__(self)
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size) / 8)
        self.scale = scale
        
    def binarize_and_smooth_labels(self, T, nb_classes, smoothing_const = 0):
        import sklearn.preprocessing
        T = T.cpu().numpy()
        T = sklearn.preprocessing.label_binarize(
            T, classes = range(0, nb_classes)
        )
        T = T * (1 - smoothing_const)
        T[T == 0] = smoothing_const / (nb_classes - 1)
        T = torch.FloatTensor(T).cuda()

        return T
        
    def pairwise_distance(self, a, squared=False):
        """Computes the pairwise distance matrix with numerical stability."""
        pairwise_distances_squared = torch.add(
            a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
            torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
        ) - 2 * (
            torch.mm(a, torch.t(a))
        )

        # Deal with numerical inaccuracies. Set small negatives to zero.
        pairwise_distances_squared = torch.clamp(
            pairwise_distances_squared, min=0.0
        )

        # Get the mask where the zero distances are at.
        error_mask = torch.le(pairwise_distances_squared, 0.0)
        #print(error_mask.sum())
        # Optionally take the sqrt.
        if squared:
            pairwise_distances = pairwise_distances_squared
        else:
            pairwise_distances = torch.sqrt(
                pairwise_distances_squared + error_mask.float() * 1e-16
            )

        # Undo conditionally adding 1e-16.
        pairwise_distances = torch.mul(
            pairwise_distances,
            (error_mask == False).float()
        )

        # Explicitly set diagonals to zero.
        mask_offdiagonals = 1 - torch.eye(
            *pairwise_distances.size(),
            device=pairwise_distances.device
        )
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

        return pairwise_distances
     
    def forward(self, X, T):
        P = self.proxies
        #note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
       
        P = self.scale * F.normalize(P, p = 2, dim = -1)
        X = self.scale * F.normalize(X, p = 2, dim = -1)
        
        XP = torch.cat([X, P])        
        D = self.pairwise_distance(XP, squared=True)[:X.size()[0], X.size()[0]:]

        T = self.binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = 0
        )

        loss = torch.sum(- T * F.log_softmax(-D, -1), -1)
        loss = loss.mean()
        return loss

class SynProxy(pl.LightningModule):
    def __init__(self, num_classes, embedding_size, scale=10.0, ps_mu=0.0, ps_alpha=0.0):
        super(SynProxy, self).__init__()
        self.scale = scale
        self.n_classes = num_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        
    def proxy_synthesis(self, input_l2, proxy_l2, target, ps_alpha, ps_mu):
        '''
        input_l2: [batch_size, dims] l2-normalized embedding features
        proxy_l2: [n_classes, dims] l2-normalized proxy parameters
        target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
        ps_alpha: alpha for beta distribution
        ps_mu: generation ratio (# of synthetics / batch_size)
        '''

        input_list = [input_l2]
        proxy_list = [proxy_l2]
        target_list = [target]

        ps_rate = np.random.beta(ps_alpha, ps_alpha)

        input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
        proxy_aug = ps_rate * proxy_l2[target,:] + (1.0 - ps_rate) * torch.roll(proxy_l2[target,:], 1, dims=0)
        input_list.append(input_aug)
        proxy_list.append(proxy_aug)
        
        n_classes = proxy_l2.shape[0]
        pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0]).cuda()
        target_list.append(pseudo_target)

        embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
        proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
        input_large = torch.cat(input_list, dim=0)[:embed_size,:]
        proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size,:]
        target = torch.cat(target_list, dim=0)[:embed_size]
    
        input_l2 = F.normalize(input_large, p=2, dim=1)
        proxy_l2 = F.normalize(proxy_large, p=2, dim=1)
        return input_l2, proxy_l2, target
    
    
    def forward(self, input, target):
        # Targets need to not be adjacent, shift them over along with inputs to match
        i = torch.cat([torch.arange(0,target.shape[0],2), torch.arange(1,target.shape[0],2)])
        input = input[i]
        target = target[i]
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)

        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = self.proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)
 
        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss
    

# Return loss_func, name, requires_optimizer
def get_ml_loss(loss, num_classes, train_ds):
    ml_losses = {
        'proxyanchor':      (pml_losses.ProxyAnchorLoss, True),
        'proxynca++':       (ProxyNCA_pp, False),
        'subcenterarcface': (pml_losses.SubCenterArcFaceLoss, True),
        'synproxy':         (SynProxy, False),
        'ntxent':           (pml_losses.NTXentLoss, False),
        'drms':             (DRMultiSimilarity, False),
        'supcon':           (pml_losses.SupConLoss, False),
        'circle':           (pml_losses.CircleLoss, False),
        'pnp':              (pml_losses.PNPLoss, False),
        'multisim':         (pml_losses.MultiSimilarityLoss, False)
    }
    
    ml_func, requires_opt = ml_losses[loss]
    if loss in ['softtriple', 'proxyanchor', 'proxynca++', 'subcenterarcface', 'synproxy']:
        initalized_func = ml_func(num_classes=num_classes, embedding_size=512)
    elif loss == 'ride':
        initalized_func = ml_func(train_ds)
    else:
        initalized_func = ml_func()
    
    return (initalized_func, requires_opt)
