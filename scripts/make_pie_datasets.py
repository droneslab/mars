import glob
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath('../src/'))
from datasets import HiRISEData, BlenderLunarHIRISEData
from utils import load_config
cfg = load_config('../cfg/laptop.yaml')
name = sys.argv[1]

if name == 'hirise':
    dataset = HiRISEData(cfg['hirise_images'], cfg['hirise_labels'], 'crater', batch_size=16)
elif name == 'lunar':
    dataset = BlenderLunarHIRISEData(cfg['lunar_images'], cfg['lunar_trajPkl'], batch_size=16)
ds = dataset.train_dataset

data = {'train': {'crater': {}}, 'test': {'crater': {}}}

l=0
for i in np.unique(ds.labels):
    c_paths = [x[0] for x in ds.instances if x[1] == i]
    if name == 'hirise':
        c_paths = [f'/mnt/Space-Vision/data/hirise_v3/map-proj-v3/{x}' for x in c_paths]
    else:
        # c_paths = [x.replace('/home/tj/data/hirise_v3/map-proj-v3/', '/mnt/Space-Vision/data/hirise_v3/map-proj-v3/') for x in c_paths]
        # pass
        print(c_paths)
        quit()
    data['train']['crater'][str(l)] = {'objs': c_paths, 'labels': l}
    l+=1

with open(f'vm_{name}.pickle', 'wb') as f:
    pickle.dump(data, f)
