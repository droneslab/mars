import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import transforms

# proxyanchor 30
cam1 = Image.open('../data/results/figs/cams/subcenterarcface/48.png')
cam2 = Image.open('../data/results/figs/cams/proxyanchor/30.png')
cam3 = Image.open('../data/results/figs/cams/proxyanchor/16.png')
map1 = Image.open('../data/results/figs/maps/cosface/2/batch1/11.png')
map2 = Image.open('../data/results/figs/maps/contrastive/2/39.png')
map3 = Image.open('../data/results/figs/maps/drms/2/2.png')

cam1 = cam1.crop((35, 0, cam1.size[0], cam1.size[1]-35))
cam2 = cam2.crop((35, 0, cam2.size[0], cam2.size[1]-35))
cam3 = cam3.crop((35, 0, cam3.size[0], cam3.size[1]-35))
map1 = map1.crop((35, 0, map1.size[0], map1.size[1]-35))
map2 = map2.crop((35, 0, map2.size[0], map2.size[1]-35))
map3 = map3.crop((35, 0, map3.size[0], map3.size[1]-35))

# cam3 = cam3.crop((35, 0, cam3.size[0], cam3.size[1]-35))
# map1 = map1.crop((35, 0, map1.size[0], map1.size[1]-35))

fig,axs = plt.subplots(3,2, figsize=(4,8))
fig.tight_layout()

axs[0][1].imshow(cam1)
axs[0][1].set_title('Attention')

axs[0][0].imshow(map1)
axs[0][0].set_title('Features')

axs[1][1].imshow(cam2)
axs[1][0].imshow(map2)
axs[2][1].imshow(cam3)
axs[2][0].imshow(map3)
plt.setp(axs, xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig(f"../data/results/figs/intro2.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)
plt.close()