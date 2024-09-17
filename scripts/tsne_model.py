# https://builtin.com/data-science/tsne-python
from __future__ import print_function
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from glob import glob
from PIL import Image
from alive_progress import alive_bar
import sys
sys.path.append('/home/tj/landmark_recognition/src/')
from model import MetricLDN
from datasets.hirise import HiRISEData
from datasets.cub_2011 import CubData
from utils import load_config, show_image_gray, getInputArguments
from alive_progress import alive_bar
import umap
import umap.plot

# args = getInputArguments(sys.argv)
# cfg = load_config(args.cfg)
# # dataset = CubData('/home/tj/data/', batch_size=8)
# dataset = HiRISEData(cfg['hirise_images'], cfg['hirise_labels'], 'crater', batch_size=args.batch_size)
# args.train_ds = dataset.train_dataset
# # model = MetricLDN.load_from_checkpoint('/home/tj/landmark_recognition/data/training_logs/latest/cub_conv_se_proxyanchor/lightning_logs/version_0/checkpoints/epoch=99-step=37400.ckpt', args=args).cuda().eval()
# model = MetricLDN.load_from_checkpoint('/home/tj/landmark_recognition/data/training_logs/latest/hirise_conv_se_proxyanchor/lightning_logs/version_0/checkpoints/epoch=99-step=17200.ckpt', args=args).cuda().eval()


# embeddings = np.zeros((len(dataset.test_dataset), 512))
# ys = np.zeros((len(dataset.test_dataset)))
# with alive_bar(len(dataset.test_dataset)) as bar:
#     for i in range(len(dataset.test_dataset)):
#         x,y,_ = dataset.test_dataset[i]
#         embedding = model(x[None,:].cuda())
#         embeddings[i,:] = embedding.detach().cpu().numpy()
#         ys[i] = y
#         bar()
        
# np.save('hirise_conv_se_proxyanchor_embs', embeddings)
# np.save('hirise_conv_se_proxyanchor_ys', ys)

# cub_xs = np.load('cub_conv_se_proxyanchor_embs.npy')
# cub_ys = np.load('cub_conv_se_proxyanchor_ys.npy')
cub_xs = np.load('hirise_conv_se_proxyanchor_embs.npy')
cub_ys = np.load('hirise_conv_se_proxyanchor_ys.npy')

# cub_xs = cub_xs[cub_ys < 25]
# cub_ys = cub_ys[cub_ys < 25]
print(cub_xs.shape)
print(cub_ys.shape)

# pca = PCA(n_components=50)
# pca_result = pca.fit_transform(cub_xs)

for p in (2,5,15,30,45,60,75,80,100):
    print('-----',p)
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=p, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(cub_xs)
    # tsne_results = tsne.fit_transform(pca_result)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    results = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two': tsne_results[:,1]}
    results['y'] = cub_ys

    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(np.unique(cub_ys))),
        data=results,
        legend=False,
        # alpha=0.75
    )
    plt.show()

# for n in (5, 10, 15, 25, 30, 45, 60, 75, 100, 150, 200):
#     mapper = umap.UMAP(n_neighbors=n, min_dist=0.001).fit(cub_xs)
#     umap.plot.points(mapper, labels=cub_ys, show_legend=False)
#     # umap.plot.connectivity(mapper, edge_bundling='hammer', labels=cub_ys)
#     plt.show()