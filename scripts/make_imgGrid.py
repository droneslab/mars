import matplotlib.pyplot as plt
from PIL import Image
ipath = '/home/tj/data/hirise_v3/map-proj-v3/'
# crater_paths = ['0.png','1726.png','1941.png','1965.png','1980.png','3174.png','3256.png','3763.png','3949.png']
crater_paths = [
    'ESP_011881_1585_RED-0068.jpg',
    'ESP_026472_1410_RED-0107.jpg',
    'ESP_016530_1465_RED-0271.jpg',
    'ESP_016730_1960_RED-0461.jpg',
    'ESP_016148_2040_RED-0073.jpg',
    'ESP_034557_1510_RED-0266.jpg',
    'ESP_013161_1715_RED-0049.jpg',
    'ESP_026334_1755_RED-0080.jpg',
    'ESP_026472_1410_RED-0155.jpg'
]
imgs = []
for cp in crater_paths:
    imgs.append( Image.open(ipath + cp) )

fig,axs = plt.subplots(3,3, figsize=(8,8))
# fig.set_size_inches(513,513)
fig.tight_layout()
axs = axs.flatten()
for img,ax in zip(imgs,axs):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=None, hspace=None)
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig("../data/results/figs/hirise_craters.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)

# c = Image.open('./craters.png')
# f = Image.open('/home/tj/data/blender/lro_trajs/stock/images/607.png')

# fig,axs = plt.subplots(1,2, figsize=(16,8))
# fig.tight_layout()
# axs = axs.flatten()
# for img,ax in zip([c,f],axs):
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(img)
# plt.show()