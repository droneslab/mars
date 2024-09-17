import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from torchvision.models.alexnet import model_urls
from torchvision.models.alexnet import model_urls as model_url_alexnet
from torchvision.models.vgg import model_urls as model_url_vgg
import torchvision.models as models
import pickle
from random import randint
import random
import numpy as np
import time
from PIL import Image


def compute_acc(feat, class_feat, gt):
    correct = 0.0
    for i in range(feat.shape[0]):
        f = feat[i].view(1,feat[i].shape[0]).repeat(class_feat.shape[0],1)
        dist = torch.sum((f - class_feat)**2,1)
        dist = dist.cpu().detach().numpy()
        pred = np.argmin(dist)
        if pred == gt[i]:
            correct += 1.0
    return correct

class Multiview_obj:
    def __init__(self, view_list, transform, num_of_view):
        # number of views
        self.V = num_of_view
        self.transform = transform
        # a list of views for a single object
        self.views = self._load_views(view_list, num_of_view)
       
    def _load_views(self, view_list, V):
        views = None

        sampled_view_list = np.random.choice(view_list, V, replace=False)
        for f in sampled_view_list:
            img = Image.open(f)
            if self.transform is not None:
                img = self.transform(img)
             
            img = img.unsqueeze(0)    
            if views is None:
                views = img
            else:
                views = torch.cat([views, img], 0)
        # views has shape [batchsize, # of view , 3, 224, 224]
        return views

class mvDataset:
    def __init__(self, data , max_view, transform = None, view_drop_out = False, sampled_view = None):
        self.data = data
        self.objs = data['objs']
        self.labels = data['labels']
        self.max_view = max_view
        self.transform = transform
        self.view_drop_out = view_drop_out
        self.sampled_view = sampled_view

    def __len__(self):
        return len(self.objs)
        
    def __preprocess__(self, views_for_a_single_object, sampled_view):
        s = Multiview_obj(views_for_a_single_object, self.transform, sampled_view)           
        return s
    
    def __getitem__(self, idx):

        objs_batch = self.objs[idx]            
        labels_batch = self.labels[idx]
        if self.view_drop_out and self.sampled_view is None:
            self.sampled_view = randint(1, self.max_view)
        else: 
            self.sampled_view = self.max_view
        
        shape_object = self.__preprocess__(objs_batch, self.sampled_view) 
        x = shape_object.views
        y = labels_batch
        return (x,y)

            
            
def load_data(pickle_filename):

    train = {}
    train['objs'] = []
    train['labels'] = []
    test = {}
    test['objs'] = []
    test['labels'] = []
    with open(pickle_filename, "rb") as input_file:
        data = pickle.load(input_file)    
    
    for c in data['train']:
        for obj in data['train'][c]:
            train['objs'].append(data['train'][c][obj]['objs'])
            train['labels'].append(data['train'][c][obj]['labels'])
            
    for c in data['test']:
        for obj in data['test'][c]:
            test['objs'].append(data['test'][c][obj]['objs'])
            test['labels'].append(data['test'][c][obj]['labels'])
            
    return train, test
    
class VGG_avg_tc(nn.Module):
    def __init__(self, input_view, output_class):
        super(VGG_avg_tc, self).__init__()
        self.input_view = input_view
        VGG = models.vgg16(pretrained=True)
        VGG.classifier._modules['6']= nn.Linear(4096, out_features=output_class)
        
        self.features = VGG.features
        self.input_view = input_view
        self.output_class = output_class                
        self.classifier1 = nn.Sequential(*list(VGG.classifier)[0:5])
        self.class_centers = nn.Parameter(torch.randn(1,1,self.output_class, 4096))
        
    def forward(self, x):
        # x: (n, self.view, 3, 224, 224)        
        image_features = None
        shape_feature = None
        for view in range(self.input_view):
            view_feature = self.features(x[:,view]).view(x.shape[0], 25088)            
            view_feature = self.classifier1(view_feature).view(x.shape[0],1,4096)

            if image_features is None:
                image_features = view_feature
            else:
                image_features = torch.cat([image_features, view_feature],1)
        
        class_feature = F.normalize(self.class_centers, p=2, dim=3, eps=1e-12)
        class_feature = class_feature.view(self.output_class,4096)
        
        image_features = image_features.view(x.shape[0]*self.input_view,-1)
        image_features = F.normalize(image_features, p=2, dim=1, eps=1e-12)
        
        return image_features, class_feature       