import torch
import torch.nn as nn
from torchvision import models
##############################################################################################################
from torch.autograd import Function
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MPNCOV(nn.Module):
    """Matrix power normalized Covariance pooling (MPNCOV)
    implementation of fast MPN-COV (i.e.,iSQRT-COV)
    https://arxiv.org/abs/1712.01034
    Args:
        iterNum: #iteration of Newton-schulz method
        is_sqrt: whether perform matrix square root or not
        is_vec: whether the output is a vector or not
        input_dim: the #channel of input feature
        dimension_reduction: if None, it will not use 1x1 conv to
                            reduce the #channel of feature.
                            if 256 or others, the #channel of feature
                            will be reduced to 256 or others.
    """
    def __init__(self, iterNum=3, is_sqrt=True, is_vec=None, input_dim=2048, dimension_reduction=None):

        super(MPNCOV, self).__init__()
        self.iterNum=iterNum
        self.is_sqrt = is_sqrt
        self.is_vec = is_vec
        self.dr = dimension_reduction
        if self.dr is not None:
            self.conv_dr_block = nn.Sequential(
            nn.Conv2d(input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.dr),
            nn.ReLU(inplace=True)
            )
        output_dim = self.dr if self.dr else input_dim
        if self.is_vec:
            self.output_dim = int(output_dim*(output_dim+1)/2)
        else:
            self.output_dim = int(output_dim*output_dim)
        self._init_weight()
 
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _cov_pool(self, x):
        return Covpool.apply(x)
    def _sqrtm(self, x):
        return Sqrtm.apply(x, self.iterNum)
    def _triuvec(self, x):
        return Triuvec.apply(x)
 
    def forward(self, x):
        if self.dr is not None:
            x = self.conv_dr_block(x)
        x = self._cov_pool(x)
        if self.is_sqrt:
            x = self._sqrtm(x)
        if self.is_vec:
            x = self._triuvec(x)
        return x
 
 
class Covpool(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize,dim,M)
        I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
        I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
        y = x.bmm(I_hat).bmm(x.transpose(1,2))
        ctx.save_for_backward(input,I_hat)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input,I_hat = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        h = x.data.shape[2]
        w = x.data.shape[3]
        M = h*w
        x = x.reshape(batchSize,dim,M)
        grad_input = grad_output + grad_output.transpose(1,2)
        grad_input = grad_input.bmm(x).bmm(I_hat)
        grad_input = grad_input.reshape(batchSize,dim,h,w)
        return grad_input
 
class Sqrtm(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
        normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize,1,1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device).type(dtype)
        Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1).type(dtype)
        if iterN < 2:
            ZY = 0.5*(I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
                ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
                Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
                Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            YZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
        y = YZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
        I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                        Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
                YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
                ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
                dldY_ = 0.5*(dldY.bmm(YZ) -
                            Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) -
                                ZY.bmm(dldY))
                dldZ_ = 0.5*(YZ.bmm(dldZ) -
                            Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i,:,:] += (der_postComAux[i] \
                                - grad_aux[i] / (normA[i] * normA[i])) \
                                *torch.ones(dim,device = x.device).diag().type(dtype)
        return grad_input, None
 
class Triuvec(Function):
    @staticmethod
    def forward(ctx, input):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        x = x.reshape(batchSize, dim*dim)
        I = torch.ones(dim,dim).triu().reshape(dim*dim)
        # index = I.nonzero()
        index = torch.nonzero(I,as_tuple=False)
        y = torch.zeros(batchSize,int(dim*(dim+1)/2),device = x.device).type(dtype)
        y = x[:,index]
        ctx.save_for_backward(input,index)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        input,index = ctx.saved_tensors
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        grad_input = torch.zeros(batchSize,dim*dim,device = x.device,requires_grad=False).type(dtype)
        grad_input[:,index] = grad_output
        grad_input = grad_input.reshape(batchSize,dim,dim)
        return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)


class SCCov_Res34(nn.Module):
    def __init__(self, clsNum, dim=128):
        super().__init__()
        
        resnet = models.resnet34(pretrained=True)

        self.conv1_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.avg2 = nn.AvgPool2d(kernel_size=(4,4))
        self.layer3 = resnet.layer3
        self.avg3 = nn.AvgPool3d(kernel_size=(2,2,2))
        self.layer4 = resnet.layer4
        self.avg4 = nn.AvgPool3d(kernel_size=(4,1,1))
        self.cov_pool = MPNCOV(is_vec=True, input_dim=384)
        self.FC1 = nn.Linear(73920, dim)
        self.FC2 = nn.Linear(dim, clsNum)

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = self.avg2(x)
        x = self.layer3(x)
        x3 =self.avg3(x)
        # print(x3.shape) 
        x = self.layer4(x)
        x4 = self.avg4(x)
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.cov_pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.FC1(x)
        logits = self.FC2(x)

        return F.normalize(x), logits

class SCCov_Res34_emb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        
        resnet = models.resnet34(pretrained=True)

        self.conv1_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.avg2 = nn.AvgPool2d(kernel_size=(4,4))
        self.layer3 = resnet.layer3
        self.avg3 = nn.AvgPool3d(kernel_size=(2,2,2))
        self.layer4 = resnet.layer4
        self.avg4 = nn.AvgPool3d(kernel_size=(4,1,1))
        self.cov_pool = MPNCOV(is_vec=True, input_dim=384)
        self.FC1 = nn.Linear(73920, dim)

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = self.avg2(x)
        x = self.layer3(x)
        x3 =self.avg3(x)
        # print(x3.shape) 
        x = self.layer4(x)
        x4 = self.avg4(x)
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.cov_pool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.FC1(x)

        return F.normalize(x)


class SCCov_Res50(nn.Module):
    def __init__(self, clsNum, dim=128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        self.conv1_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.avg2 = nn.AvgPool3d(kernel_size=(4,4,4))
        self.layer3 = resnet.layer3
        self.avg3 = nn.AvgPool3d(kernel_size=(8,2,2))
        self.layer4 = resnet.layer4
        self.avg4 = nn.AvgPool3d(kernel_size=(16,1,1))
        self.cov_pool = MPNCOV(is_vec=True, input_dim=384)
        self.FC1 = nn.Linear(73920, dim)
        self.FC2 = nn.Linear(dim, clsNum)
        
    def forward(self, x):
        x = self.conv1_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = self.avg2(x)
        x = self.layer3(x)
        x3 =self.avg3(x)
        x = self.layer4(x)
        x4 = self.avg4(x)
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.cov_pool(x)
        x = x.view(x.size(0), -1)

        x = self.FC1(x)
        logits = self.FC2(x)

        return F.normalize(x), logits

class SCCov_Res50_emb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        self.conv1_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.avg2 = nn.AvgPool3d(kernel_size=(4,4,4))
        self.layer3 = resnet.layer3
        self.avg3 = nn.AvgPool3d(kernel_size=(8,2,2))
        self.layer4 = resnet.layer4
        self.avg4 = nn.AvgPool3d(kernel_size=(16,1,1))
        self.cov_pool = MPNCOV(is_vec=True, input_dim=384)
        self.FC1 = nn.Linear(73920, dim)

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = self.avg2(x)
        x = self.layer3(x)
        x3 =self.avg3(x)
        x = self.layer4(x)
        x4 = self.avg4(x)
        x = torch.cat((x2, x3, x4), dim=1)
        x = self.cov_pool(x)
        x = x.view(x.size(0), -1)
        x = self.FC1(x)

        return F.normalize(x)

if __name__ == '__main__':

    x = torch.randn((1,3,256,256))

    model = SCCov_Res50(10)

    model(x)




