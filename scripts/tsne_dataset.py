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

# image_paths = sorted(glob('/home/tj/data/CUB_200_2011/images/**/*.jpg'))
# xs = np.zeros((len(image_paths),28,28))
# ys = np.zeros((len(image_paths)))
# with alive_bar(len(image_paths)) as bar:
#     for i in range(len(image_paths)):
#         path = image_paths[i]
#         lbl = int(path.split('/')[-2].split('.')[0])
#         img = Image.open(path).convert('L')
#         data = np.asarray(img)
#         xs[i,:] = data
#         ys[i] = lbl
#         bar()
# np.save('cub_200_2011_imgs', xs)
# np.save('cub_200_2011_lbls', ys)

cubs_xs = np.load('cub_200_2011_imgs.npy')
cubs_ys = np.load('cub_200_2011_lbls.npy')

cubs_xs = cubs_xs.reshape(cubs_xs.shape[0], cubs_xs.shape[1]*cubs_xs.shape[2])

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(cubs_xs)

pca = PCA(n_components=50)
pca_result = pca.fit_transform(cubs_xs)

time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(cubs_xs)
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pca_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

results = {'tsne-2d-one': tsne_results[:,0], 'tsne-2d-two': tsne_results[:,1]}
results['y'] = cubs_ys

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 200),
    data=results,
    legend="brief",
    alpha=0.3
)
plt.show()