import sys
sys.path.append('../src/')
from datasets import HiRISEData
from utils import load_config, show_image_gray
cfg = load_config("../cfg.yaml")
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from alive_progress import alive_bar
from geoRender import parseModelCkpts
from model import ResNext
import torch
'''
1. Test image, generate scale/rotating set based on params in paper
2. Take image, get embedding
3. For each aug image, get embedding, cosine distance
4. Create plot like in paper
'''
            
def collectDistances(x,rots,scales,model):
    stock_emb = model(x[None,:])
    # dists = []
    dists = np.zeros((len(rots),len(scales)))
    imgs = []
    grids = [(-165,67), (-165,100), (-165,133),
             (0,67), (0,100), (0,133),
             (165,67), (165,100), (165,133)]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    with alive_bar(len(rots)*len(scales)) as bar:
        # for rot,scale in combos:
        for r in range(len(rots)):
            rot = rots[r]
            for s in range(len(scales)):
                scale = scales[s]
                aug = torchvision.transforms.functional.affine(x,translate=(0,0),shear=0,angle=rot,scale=(scale/100))
                if (rot,scale) in grids:
                    imgs.append(aug)
                aug_emb = model(aug[None,:])
                sim = cos(stock_emb,aug_emb)
                # dists.append((rot,scale,sim.detach().cpu().item()))
                dist = sim.detach().cpu().item()
                dists[s][r] = dist
                bar()
    return dists, imgs

ds = HiRISEData(cfg['hirise_images'], cfg['hirise_labels'], shuffle=True)
test_ds = ds.test_dataset
x,y = test_ds[0]
x = test_ds.get_stock_img(y).cuda()
rots = list(range(-165,180,15))
scales = list(range(67,136,3))
# combos = [(x,y) for x in rots for y in scales]

models = parseModelCkpts(ckpt_path='../src/ccr_logs/30epochs_OLD/**/**/**/checkpoints/', type='ckpt-epoch=29')
types = ['ArcFace', 'CosFace', 'SubCenterArcFace', 'ProxyNCA', 'ProxyAnchor', 'Contrastive', 'LiftedStructure', 'NTXent']
grids = []
dists = []
for model_type in types:
    print(model_type)
    model = ResNext.load_from_checkpoint(models[model_type]['path'], bneck_type='se', loss_func=model_type.lower(), miner=models[model_type]['miner']).cuda().eval()
    d,imgs = collectDistances(x,rots,scales, model)
    img_grid = torchvision.utils.make_grid(imgs, nrow=3)
    grids.append(img_grid)
    
    dists.append((model_type, d))

    # fig = plt.figure()
    # ax2 = fig.add_subplot(1,2,1)
    # ax2 = fig.add_subplot(1,2,2, projection='3d')
    # ax2 = fig.add_subplot(1,2,2)
    # ax1.imshow(img_grid.permute(1, 2, 0).detach().cpu().numpy())
    # ax2.plot_surface(X, Y, dists, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    # cont = ax1.contourf(X,Y,dists, 100, cmap='RdGy', vmin=0, vmax=1)
    # ax2.colorbar()
    # plt.colorbar(cont, ax=ax2)
    # ax2.set_xlabel('Rotation Angle')
    # ax2.set_ylabel('Scale Percent')
    # ax2.set_zlabel('Cosine Similarity')
    # plt.show()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1,1,1, projection='3d')
    # cont = ax1.contourf(X,Y,dists, 100, cmap='RdGy', vmin=0, vmax=1)
    # ax1.set_xlabel('Rotation Angle')
    # ax1.set_ylabel('Scale Percent')

X,Y = np.meshgrid(rots,scales)

fig = plt.figure()
ax1 = fig.add_subplot(2,4,1)
ax2 = fig.add_subplot(2,4,2)
ax3 = fig.add_subplot(2,4,3)
ax4 = fig.add_subplot(2,4,4)
ax5 = fig.add_subplot(2,4,5)
ax6 = fig.add_subplot(2,4,6)
ax7 = fig.add_subplot(2,4,7)
ax8 = fig.add_subplot(2,4,8)
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
for i in range(len(axes)):
    m = dists[i]
    name = m[0]
    dist = m[1]
    ax = axes[i]
    cont = ax.contourf(X,Y,dist, 100, cmap='RdGy', vmin=0, vmax=1)
    # plt.colorbar(cont, ax=ax)
    ax.set_xlabel('Rotation Angle')
    ax.set_ylabel('Scale Percent')
    ax.set_title(name)
fig2 = plt.figure(2)
plt.imshow(grids[0].permute(1, 2, 0).detach().cpu().numpy())
plt.show()