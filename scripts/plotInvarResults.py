import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import pandas as pd

csv_files = glob.glob('../results/30eps_ranInvar/*.csv')
data = {}
for csv_file in csv_files:
    name = csv_file.split('.csv')[0]
    name = name.split('/')[-1]
    df = pd.read_csv(csv_file)
    data[name] = df
plot_order = ['CosFace', 'ArcFace', 'SubCenterArcFace', 'Contrastive', 'TripletMargin', 'LiftedStructure', 'NTXent', 'ProxyNCA', 'ProxyAnchor']

# plot_order = ['CosFace', 'ArcFace', 'SubCenterArcFace', 'Contrastive', 'LiftedStructure', 'NTXent', 'ProxyNCA', 'ProxyAnchor', 'RiDe_resnet34', 'RiDe_resnet50', 'RiDe_bninception']

means = {}
stds = {}
for model in plot_order:
    means[model] = {}
    stds[model] = {}
    df = data[model]
    
    for column in df:
        means[model][column] = df[column].mean()
        stds[model][column] = df[column].std()
    
# bright_cols = ['bright25', 'dark25', 'bright50', 'dark50', 'bright75', 'dark75']
# bright_cols = ['bright25', 'bright50', 'bright75', 'dark25', 'dark50', 'dark75']
# rot_cols = ['rot90', 'rot180', 'rot270', 'rot1', 'rot2', 'rot3']
# tp_noise_cols = ['trans', 'pers', 'gauss', 'shot']

labels = ['ill','trans','rot','pers']

# model_cols = ['firebrick', 'lightcoral', 'r', 'dodgerblue', 'skyblue', 'c', 'g', 'darkseagreen']
# cmap = ['firebrick', 'dodgerblue', 'darkseagreen']
cmap = plt.cm.get_cmap('tab20c')
colors = [cmap(i) for i in [0,1,2,4,5,6,7,8,9]]
# colors = [cmap(i) for i in [0,1,2,4,5,6,8,9,12,13,14]]
# colors = [cmap(i) for i in [0,0,0,1,1,1,2,2,2]]

# for labels in [bright_cols, rot_cols, tp_noise_cols]: 
fig, ax = plt.subplots()
width = 0.075
x = np.arange(len(labels))
bws = np.linspace(-0.25, 0.25, len(plot_order))

y=0
for model in plot_order:
    # rect = ax.bar(x+bws[y], list(means[model].values())[:len(labels)], width, label=model, color=model_cols[y], yerr=list(stds[model].values())[:len(labels)])
    rect = ax.bar(x+bws[y], [means[model][x] for x in labels], width, label=model, color=colors[y], yerr=[stds[model][x] for x in labels])
    ax.bar_label(rect, fmt='%.3f', padding=3)
    y+=1
ax.set_ylabel('Cosine Similarity')
ax.set_xticks(x, ['Illumination', 'Translation', 'Rotation', 'Perspective'])
ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=6)
plt.axhline(y=0.9, color='k', alpha=0.2, linestyle='--')
plt.show()
plt.close()