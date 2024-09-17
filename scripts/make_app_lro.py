import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2

with open('/home/tj/data/blender/lro_trajs/stock/annotations.txt', 'r') as f:
    lines = f.readlines()
np.random.shuffle(lines)
np.random.shuffle(lines)
np.random.shuffle(lines)
np.random.shuffle(lines)
lines = [l.strip() for l in lines]
lines = lines[:9]

imgs = []
boxed = []
for l in lines:
    vs = l.split(' ')
    ipath = vs[0]
    imgs.append(Image.open(ipath))
    boxes = vs[1:] # each box is in l,r,t,b format
    img = cv2.imread(ipath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for b in boxes:
        bv = b.split(',')
        id, l,r,t,b = bv
        l,r,t,b = [int(x) for x in [l,r,t,b]]
        cv2.rectangle(img, (l,t), (r,b), (249,114,114), 2)
    boxed.append(img)

fig,axs = plt.subplots(3,3, figsize=(12,12))
fig.tight_layout()
axs = axs.flatten()
for img,ax in zip(imgs,axs):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig("../data/results/figs/lunar_app_lroImgGrid.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)

fig,axs = plt.subplots(3,3, figsize=(12,12))
fig.tight_layout()
axs = axs.flatten()
for img,ax in zip(boxed,axs):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig("../data/results/figs/lunar_app_lroImgBox.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)