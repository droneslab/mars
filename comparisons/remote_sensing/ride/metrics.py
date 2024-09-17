
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report

class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class KNNClassification(nn.Module):

    def __init__(self, X_train, Y_true, K=10):
        super().__init__()

        self.K = K

        self.KNN = KNeighborsClassifier(n_neighbors=self.K, n_jobs=-1)
        self.KNN.fit(X_train, Y_true)

    def forward(self, X_test, y_true):

        y_pred = self.KNN.predict(X_test)

        acc = accuracy_score(y_true, y_pred)

        return acc


class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



class HingeLoss(nn.Module):
    """
    Hinge loss based on the paper:
    when deep learning meets metric learning:remote sensing image scene classification
    via learning discriminative CNNs 
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/9
    """

    def __init__(self, margin=0.44):
        super().__init__()
        
        self.margin = margin

    def forward(self, oneHotCodes, features):
        
        L_S = oneHotCodes.mm(torch.t(oneHotCodes))
        Dist = torch.norm(features[:,None] - features, dim=2, p=2)**2

        Dist = self.margin - Dist
        
        L_S[L_S==0] = -1

        Dist = 0.05 - L_S * Dist

        loss = torch.triu(Dist, diagonal=1)

        loss[loss < 0] = 0

        return torch.mean(loss)





class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim,
                 num_instances,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        if instance_targets is not None:
            loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
            return prediction_logits, loss
        else:
            return prediction_logits


class NormSoftmaxLoss_Margin(nn.Module):
    """ 
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """
    def __init__(self,
                 dim,
                 num_instances,
                 margin=0.5,
                 temperature=0.05):
        super().__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings, label):
        norm_weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, norm_weight)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        loss = self.loss_fn(logits / self.temperature, label)
        return logits, loss




def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y

def compute_aff(x):
    """computes the affinity matrix between an input vector and itself"""
    return torch.mm(x, x.t())

class SmoothAP(torch.nn.Module):
    """PyTorch implementation of the Smooth-AP loss.
    implementation of the Smooth-AP loss. Takes as input the mini-batch of CNN-produced feature embeddings and returns
    the value of the Smooth-AP loss. The mini-batch must be formed of a defined number of classes. Each class must
    have the same number of instances represented in the mini-batch and must be ordered sequentially by class.
    e.g. the labels for a mini-batch with batch size 9, and 3 represented classes (A,B,C) must look like:
        labels = ( A, A, A, B, B, B, C, C, C)
    (the order of the classes however does not matter)
    For each instance in the mini-batch, the loss computes the Smooth-AP when it is used as the query and the rest of the
    mini-batch is used as the retrieval set. The positive set is formed of the other instances in the batch from the
    same class. The loss returns the average Smooth-AP across all instances in the mini-batch.
    Args:
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function. A low value of the temperature
            results in a steep sigmoid, that tightly approximates the heaviside step function in the ranking function.
        batch_size : int
            the batch size being used during training.
        num_id : int
            the number of different classes that are represented in the batch.
        feat_dims : int
            the dimension of the input feature embeddings
    Shape:
        - Input (preds): (batch_size, feat_dims) (must be a cuda torch float tensor)
        - Output: scalar
    Examples::
        >>> loss = SmoothAP(0.01, 60, 6, 256)
        >>> input = torch.randn(60, 256, requires_grad=True).cuda()
        >>> output = loss(input)
        >>> output.backward()
    """

    def __init__(self, anneal, batch_size, num_id, feat_dims):
        """
        Parameters
        ----------
        anneal : float
            the temperature of the sigmoid that is used to smooth the ranking function
        batch_size : int
            the batch size being used
        num_id : int
            the number of different classes that are represented in the batch
        feat_dims : int
            the dimension of the input feature embeddings
        """
        super(SmoothAP, self).__init__()

        assert(batch_size%num_id==0)

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        """Forward pass for all input predictions: preds - (batch_size x feat_dims) """

        # ------ differentiable ranking of all retrieval set ------
        # compute the mask which ignores the relevance score of the query to itself
        mask = 1.0 - torch.eye(self.batch_size) 
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.cuda()
        # compute the rankings
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        # ------ differentiable ranking of only positive set in retrieval set ------
        # compute the mask which only gives non-zero weights to the positive set
        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = pos_mask.unsqueeze(dim=0).unsqueeze(dim=0).repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        # compute the relevance scores
        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(1, 1, int(self.batch_size / self.num_id), 1)
        # compute the difference matrix
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        # pass through the sigmoid
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.cuda()
        # compute the rankings of the positive set
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        # sum the values of the Smooth-AP for all instances in the mini-batch
        ap = torch.zeros(1).cuda()
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(sim_pos_rk[ind] / (sim_all_rk[(ind * group):((ind + 1) * group), (ind * group):((ind + 1) * group)]))
            ap = ap + ((pos_divide / group) / self.batch_size)

        return (1-ap)
