import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy.misc
from PIL import Image
import json
import glob
import sys
sys.path.append('../src/')
from datasets import HIRISEDataset
from utils import load_config
CFG = load_config("../cfg.yaml")
from model import FeatureExtractor
from utils import show_image_gray
import layers.resnext as resnext
import cv2
from sklearn.manifold import TSNE
import umap
import umap.plot

model_ckpts = glob.glob('../src/ccr_logs/30epochs/**/**/**/checkpoints/ckpt-epoch=29.ckpt')
dataset = HIRISEDataset(CFG['image_dir'], CFG['label_file'], test=True, shuffle=True)

def comparitive():
    for i in range(10):
        processed = []
        names = []
        for ckpt_path in model_ckpts:
            name = ckpt_path.split('SENet_')[1]
            name = name.split('-')[0]
            name = name.split('Loss')[0]
            name = name.lower()
                
            name = 'subcenter_arcface' if name == 'subcenterarcface' else name
            name = 'lifted_structure' if name == 'liftedstructure' else name
            miner = 'multisim' if 'MultiSimilarityMiner' in ckpt_path else 'none'
            
            names.append(name)
            
            print(f'----- {name} -----')
            
            model = FeatureExtractor.load_from_checkpoint(ckpt_path, loss_func=name, miner=miner).cuda()
            model.eval()
            
            x,y = dataset[i]
            x = x.cuda()
            
            outputs = model.resnext.get_all_features(x[None,:])
            
            processed.append(x.permute(1, 2, 0).cpu().numpy())
            for feature_map in outputs:
                feature_map = feature_map.squeeze(0)
                gray_scale = torch.sum(feature_map,0)
                gray_scale = gray_scale / feature_map.shape[0]
                processed.append(gray_scale.data.cpu().numpy())
                
        # fig = plt.figure()
        fig, axs = plt.subplots(6,7, squeeze=False)
        for i in range(6):
            axs[i,0].text(0,0.5, names[i], fontsize=14)
            axs[i,0].axis("off")
        cnt = 0
        for row in range(6):
            for col in range(6):
                axs[row,col+1].imshow(processed[cnt])
                cnt += 1
                axs[row,col+1].axis("off")
        fig.tight_layout(pad=0)
        
        # Get the bounding boxes of the axs including text decorations
        import matplotlib.transforms as mtrans
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)

        #Get the minimum and maximum extent, get the coordinate half-way between those
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axs.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axs.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        y = ys[2]
        line = plt.Line2D([0,1],[y,y], transform=fig.transFigure, color="black", alpha=0.5, linestyle='--')
        fig.add_artist(line)
        
        plt.show()
        
def instance():
    for _ in range(10):
        x,y = dataset[_]
        cl = y
        batch_imgs = dataset.get_all_instances(y)
        processed = []
        names = []
        # embs = {}
        embs = []
        for ckpt_path in model_ckpts:
            name = ckpt_path.split('SENet_')[1]
            name = name.split('-')[0]
            name = name.split('Loss')[0]
            name = name.lower()
            name = 'subcenter_arcface' if name == 'subcenterarcface' else name
            name = 'lifted_structure' if name == 'liftedstructure' else name
            miner = 'multisim' if 'MultiSimilarityMiner' in ckpt_path else 'none'
            print(f'----- {name} -----')
            names.append(name)
            model = FeatureExtractor.load_from_checkpoint(ckpt_path, loss_func=name, miner=miner).cuda()
            model.eval()        
            
            # embs[name] = []
            for i in range(7):
                x = batch_imgs[i,:]
                x = x.cuda()
                outputs = model.resnext.get_all_features(x[None,:])
                processed.append(x.permute(1, 2, 0).cpu().numpy())
                for feature_map in outputs:
                    feature_map = feature_map.squeeze(0)
                    gray_scale = torch.sum(feature_map,0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    processed.append(gray_scale.data.cpu().numpy())
                emb = model(x[None,:])
                # embs[name].append(emb.detach().cpu().numpy())
                embs.append(emb.detach().cpu().numpy())
                
            # embeddings = np.array(embeddings).squeeze()
            # embs_projected = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=6).fit_transform(embeddings)
            # print(embs_projected.shape)
            # plt.scatter(embs_projected[:,0], embs_projected[:,1])
            # plt.show()
        
        # snes = {}
        # for name in list(embs.keys()):
        #     n_embs = np.array(embs[name]).squeeze()
        #     snes[name] = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=6).fit_transform(n_embs)
        
        # snes = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=6).fit_transform(np.array(embs).squeeze())
        # plt.figure()
        # for i in range(len(names)):
        #     start = i*7
        #     stop = start+7
        #     plt.scatter(snes[start:stop, 0], snes[start:stop, 1], label=names[i])
        # plt.legend()
        # plt.show()
        
        ys = np.repeat(names, 7)
        mapper = umap.UMAP(n_neighbors=15, min_dist=0.25).fit(np.array(embs).squeeze())
        umap.plot.points(mapper, labels=ys, theme='fire', show_legend=True)
        plt.tight_layout(pad=0)
        plt.savefig(f'../data/feature_plots/instance_umap_{cl}.jpg')
            
                        
        fig, axs = plt.subplots(7*2,6*3, squeeze=False)
        cnt = 0
        for row in range(7*2):
            for col in range(6*3):
                axs[row,col].imshow(processed[cnt])
                cnt += 1
                axs[row,col].axis("off")
        import matplotlib.transforms as mtrans
        r = fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, axs.flat)), mtrans.Bbox).reshape(axs.shape)
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axs.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axs.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)
        y = ys[6]
        line = plt.Line2D([0,1],[y,y], transform=fig.transFigure, color="black", alpha=0.5, linestyle='--')
        fig.add_artist(line)        
        fig.set_size_inches((20, 12), forward=False)
        plt.savefig(f'../data/feature_plots/instance_{cl}.jpg')
        
        
        
    
        
if __name__ == '__main__':
    # comparitive()
    instance()