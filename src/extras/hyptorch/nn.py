import math
import torch
import torch.nn as nn
import torch.nn.init as init
from .pmath import *


class HyperbolicMLR(nn.Module):
    r"""
    Module which performs softmax classification
    in Hyperbolic space.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = expmap0(self.p_vals, c=c)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = _hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits

    def extra_repr(self):
        return "Poincare ball dim={}, n_classes={}, c={}".format(
            self.ball_dim, self.n_classes, self.c
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))

class HypLinear(nn.Module):
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return project(mv, c=c)
        else:
            bias = expmap0(self.bias, c=c)
            return project(mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, c={}".format(
            self.in_features, self.out_features, self.bias is not None, self.c
        )
        
class HypDistanceLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, c, train_c=False, train_x=False, riemannian=False, clip_r=None):
        super(HypDistanceLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.to_poincare = ToPoincare(c, train_c, train_x, in_features, riemannian, clip_r)
        self.c = self.to_poincare.c
        self.w_scale = 1
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, x, c=None):
        # maxnorm = (1 - 1e-5) / (self.c ** 0.5)
        # w_norm = torch.norm(self.weight, dim=-1, keepdim=True) + 1e-5
        # weight = self.to_poincare(self.weight / w_norm * maxnorm * self.w_scale)
        
        # weight = F.normalize(self.weight, dim=-1)
        weight = self.to_poincare(self.weight)
        mv = dist_matrix(x, weight, c=self.c)
        return -mv

def parent_order_penalty_cdist(parent, child, mrg, c=None):
    """Penalty for parents to have smaller norm than children."""
    return torch.clip(_dist0(parent, c=c, keepdim=True).t() - _dist0(child, c=c, keepdim=True) + mrg, 0) + 1.0
    
class HypHCLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, c, train_c=False, train_x=False, riemannian=False, clip_r=None):
        super(HypHCLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.to_poincare = ToPoincare(c, train_c, train_x, in_features, riemannian, clip_r)
        self.c = self.to_poincare.c
        self.gamma = 0.25
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
                
    def p_par_to_broadcast(self, x, weight):
        weight = self.to_poincare(self.weight)
        dist = dist_matrix(x, weight, c=self.c)
        res = dist * parent_order_penalty_cdist(weight, x, self.gamma, c=self.c)
        return res
            
    def forward(self, x, c=None):
        dist = self.p_par_to_broadcast(x, self.weight)
        return dist
    
        
class ConcatPoincareLayer(nn.Module):
    def __init__(self, d1, d2, d_out, c):
        super(ConcatPoincareLayer, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out

        self.l1 = HypLinear(d1, d_out, bias=False, c=c)
        self.l2 = HypLinear(d2, d_out, bias=False, c=c)
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return mobius_add(self.l1(x1), self.l2(x2), c=c)

    def extra_repr(self):
        return "dims {} and {} ---> dim {}".format(self.d1, self.d2, self.d_out)


class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return "c={}".format(self.c)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = RiemannianGradient
        self.riemannian.c = c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
            
        if self.train_x:
            xp = project(expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(project(expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(project(expmap0(x, c=self.c), c=self.c))
        
    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)


class FromPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Poincare ball
    to n-dim Euclidean space
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):

        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        if self.train_x:
            xp = project(expmap0(self.xp, c=self.c), c=self.c)
            return logmap(xp, x, c=self.c)
        return logmap0(x, c=self.c)

    def extra_repr(self):
        return "train_c={}, train_x={}".format(self.train_c, self.train_x)
    