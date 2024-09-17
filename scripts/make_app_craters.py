import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
ipaths = glob.glob('/home/tj/data/blender/lunar_craters/raw/*.png')
np.random.shuffle(ipaths)
ipaths = ipaths[:36]

imgs = []
for cp in ipaths:
    imgs.append( Image.open(cp) )

fig,axs = plt.subplots(6,6, figsize=(12,12))
fig.tight_layout()
axs = axs.flatten()
for img,ax in zip(imgs,axs):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig("../data/results/figs/lunar_app_craterGrid.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)