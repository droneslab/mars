import wandb
import sys
import os
sys.path.append(os.path.abspath('../src/'))
from model import MetricLDN
import argparse
from datasets import BlenderLunarHIRISEData
from utils import load_config, show_image_gray
from pathlib import Path
import torchvision as tv
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import cv2
import glob
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import numpy as np

def normalize_image(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def warp(tensor, ttype, rot='+'):
    if 'r' in ttype:
        ang = int(ttype.split('r')[-1])
        if rot == '-':
            ang = -ang
        return tv.transforms.functional.rotate(tensor, ang)
    elif ttype == 'vf':
        return tv.transforms.functional.vflip(tensor)
    elif ttype == 'hf':
        return tv.transforms.functional.hflip(tensor)
    else:
        return tensor

img_paths = sorted(glob.glob('/home/tj/data/blender/lunar_craters/hirise_style/*.png'))
img_paths = [x for x in img_paths if 'vf' in x or 'r270' in x]

paired_paths = []
for i in range(0, len(img_paths), 2):
    ipath1 = img_paths[i]
    ipath2 = img_paths[i+1]
    paired_paths.append((ipath1, ipath2))
idxs = np.random.randint(0, len(paired_paths), 50)

stock = sys.argv[1]
ours = sys.argv[2]
mname = sys.argv[3]
Path(f'../data/results/figs/maps/{mname}/').mkdir(parents=True, exist_ok=True)
Path(f'../data/results/figs/maps/{mname}/0').mkdir(parents=True, exist_ok=True)
Path(f'../data/results/figs/maps/{mname}/1').mkdir(parents=True, exist_ok=True)
Path(f'../data/results/figs/maps/{mname}/2').mkdir(parents=True, exist_ok=True)
Path(f'../data/results/figs/maps/{mname}/3').mkdir(parents=True, exist_ok=True)

cfg = load_config('../cfg/laptop.yaml')
ds = BlenderLunarHIRISEData(cfg['lunar_images'], cfg['lunar_trajPkl'], batch_size=16)

stock_model_path = f'model-{stock}:v0'
our_model_path = f'model-{ours}:v0'

wb = wandb.init(project='corl_23', id=stock, resume='must', dir='/home/tj/projects/research/landmark_recognition/wandb/')

path = wandb.use_artifact(f'tjchase34/corl_23/{stock_model_path}').download('/home/tj/projects/research/landmark_recognition/artifacts/')
args = argparse.Namespace(attention='se', fa_loss='none', ma_loss='none', ml_loss=mname, caml_theta=0.99, train_ds=ds.train_dataset)
stock_model = MetricLDN.load_from_checkpoint(Path(path) / "model.ckpt", args=args).eval()

path = wandb.use_artifact(f'tjchase34/corl_23/{our_model_path}').download('/home/tj/projects/research/landmark_recognition/artifacts/')
args = argparse.Namespace(attention='se', fa_loss='caml', ma_loss='aws', ml_loss=mname, caml_theta=0.99, train_ds=ds.train_dataset)
our_model = MetricLDN.load_from_checkpoint(Path(path) / "model.ckpt", args=args).eval()

transform = transforms.Compose([
    transforms.Resize((128,128), antialias=True),
    transforms.Lambda(normalize_image)
])

n = 0
# for i in range(0, len(img_paths), 2):
for idx in idxs:
    paths = paired_paths[idx]
    ipath1 = paths[0]
    ipath2 = paths[1]
    
    img1 = read_image(ipath1, mode=ImageReadMode.RGB).float()
    img1 = transform(img1)
    t1 = ipath1.split('/')[-1].split('_')[-1].split('.png')[0]
    
    img2 = read_image(ipath2, mode=ImageReadMode.RGB).float()
    img2 = transform(img2)
    t2 = ipath2.split('/')[-1].split('_')[-1].split('.png')[0]
    
    so1 = stock_model.features.get_all_features(img1[None,:])
    oo1 = our_model.features.get_all_features(img1[None,:])
    so2 = stock_model.features.get_all_features(img2[None,:])
    oo2 = our_model.features.get_all_features(img2[None,:])
    
    img1_norm = warp(img1, t1, rot='+')
    img2_norm = warp(img2, t2, rot='+')
    
    so1_cam_imgs = []
    for feature in so1:
        nd = warp(feature, t1, rot='+')
        nd = nd.squeeze(0)
        gray_scale = torch.sum(nd,0)
        gray_scale = gray_scale / nd.shape[0]
        cam = tv.transforms.functional.resize(gray_scale[None,:], (128,128))
        cam = -cam
        vis = show_cam_on_image(img1_norm.permute(1, 2, 0).detach().cpu().numpy(), cam[0,:].detach().cpu().numpy(), use_rgb=True, image_weight=0.7, colormap=cv2.COLORMAP_JET)
        so1_cam_imgs.append(vis)
        
    so2_cam_imgs = []
    for feature in so2:
        nd = warp(feature, t2, rot='+')
        nd = nd.squeeze(0)
        gray_scale = torch.sum(nd,0)
        gray_scale = gray_scale / nd.shape[0]
        cam = tv.transforms.functional.resize(gray_scale[None,:], (128,128))
        cam = -cam
        vis = show_cam_on_image(img2_norm.permute(1, 2, 0).detach().cpu().numpy(), cam[0,:].detach().cpu().numpy(), use_rgb=True, image_weight=0.7, colormap=cv2.COLORMAP_JET)
        so2_cam_imgs.append(vis)
        
    oo1_cam_imgs = []
    for feature in oo1:
        nd = warp(feature, t1, rot='+')
        nd = nd.squeeze(0)
        gray_scale = torch.sum(nd,0)
        gray_scale = gray_scale / nd.shape[0]
        cam = tv.transforms.functional.resize(gray_scale[None,:], (128,128))
        cam = -cam
        vis = show_cam_on_image(img1_norm.permute(1, 2, 0).detach().cpu().numpy(), cam[0,:].detach().cpu().numpy(), use_rgb=True, image_weight=0.7, colormap=cv2.COLORMAP_JET)
        oo1_cam_imgs.append(vis)
        
    oo2_cam_imgs = []
    for feature in oo2:
        nd = warp(feature, t2, rot='+')
        nd = nd.squeeze(0)
        gray_scale = torch.sum(nd,0)
        gray_scale = gray_scale / nd.shape[0]
        cam = tv.transforms.functional.resize(gray_scale[None,:], (128,128))
        cam = -cam
        vis = show_cam_on_image(img2_norm.permute(1, 2, 0).detach().cpu().numpy(), cam[0,:].detach().cpu().numpy(), use_rgb=True, image_weight=0.7, colormap=cv2.COLORMAP_JET)
        oo2_cam_imgs.append(vis)
        
    for i in range(4):
        so1i = so1_cam_imgs[i]
        so2i = so2_cam_imgs[i]
        oo1i = oo1_cam_imgs[i]
        oo2i = oo2_cam_imgs[i]
        fig,axs = plt.subplots(3,2, figsize=(4,5.4))
        fig.tight_layout()
        axs[0][0].imshow(img1.permute(1,2,0))
        axs[0][1].imshow(img2.permute(1,2,0))
        axs[1][0].imshow(so1i)
        axs[1][1].imshow(so2i)
        axs[2][0].imshow(oo1i)
        axs[2][1].imshow(oo2i)
        plt.setp(axs, xticks=[], yticks=[])
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        # plt.show()
        plt.savefig(f"../data/results/figs/maps/{mname}/{i}/{n}.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)
        plt.close()
    n += 1
    
    