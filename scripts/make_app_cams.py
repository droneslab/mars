import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
# ipath = '../data/results/figs/cams/'
# arcs = ['arcface/0.png', 'arcface/4.png', 'arcface/8.png']
# prox = ['proxyanchor/0.png', 'proxyanchor/17.png', 'proxyanchor/34.png']
# lift = ['liftedstructure/7.png', 'liftedstructure/10.png', 'liftedstructure/12.png']
# arc_imgs = []
# for cp in arcs:
#     arc_imgs.append( Image.open(ipath + cp) )
    
# prox_imgs = []
# for cp in prox:
#     prox_imgs.append( Image.open(ipath + cp) )

# lift_imgs = []
# for cp in lift:
#     lift_imgs.append( Image.open(ipath + cp) )

ipath = '../data/results/figs/maps/'
one = ['drms/2/3.png', 'drms/2/5.png', 'drms/2/23.png']
two = ['cosface/2/3.png', 'cosface/2/6.png', 'cosface/2/29.png']
thr = ['proxynca/2/30.png', 'proxynca/2/40.png', 'proxynca/2/43.png']

imgs = []
for cp in one:
    imgs.append( Image.open(ipath + cp) )
for cp in two:
    imgs.append( Image.open(ipath + cp) )
for cp in thr:
    imgs.append( Image.open(ipath + cp) )

fig,axs = plt.subplots(3,3, figsize=(8,11))
fig.tight_layout()

axs[0][0].set_title('DR-MS')
axs[0][1].set_title('CosFace')
axs[0][2].set_title('Proxy NCA')

axs[0][0].imshow(imgs[0])
axs[1][0].imshow(imgs[1])
axs[2][0].imshow(imgs[2])
# axs[3][0].imshow(arc_imgs[3])

axs[0][1].imshow(imgs[3])
axs[1][1].imshow(imgs[4])
axs[2][1].imshow(imgs[5])
# axs[3][1].imshow(prox_imgs[3])

axs[0][2].imshow(imgs[6])
axs[1][2].imshow(imgs[7])
axs[2][2].imshow(imgs[8])
# axs[3][2].imshow(lift_imgs[3])


plt.setp(axs, xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
# plt.show()
plt.savefig("../data/results/figs/app_map.png",
            bbox_inches ="tight",
            pad_inches = 0,
            transparent = True,
            dpi=800)