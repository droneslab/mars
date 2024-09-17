import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl
import numpy as np
from glob import glob
import torchvision.transforms.functional as TF
import pickle
from datasets.transforms import get_transform, normalize_image
from datasets.sampler import CustomMPerClassSampler

class CommonDataset(Dataset):
    def __init__(self, instances, transform_type='none', img_sz=128):
        self.instances = instances
        self.labels = [x[1] for x in self.instances]
        self.num_classes = len(np.unique(self.labels))
        self.transform_type = transform_type
        self.img_sz = img_sz
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        path, lbl = self.instances[idx]
        img = read_image(path, mode=ImageReadMode.RGB).float()
        img = normalize_image(img)
        img = TF.resize(img, self.img_sz, antialias=True)
        
        F, T128, T32, T16, T8, T4 = get_transform(self.transform_type)
                
        return F(img), lbl, T128,T32,T16,T8,T4
    
class CommonDatamodule(pl.LightningDataModule):
    def __init__(self, image_dir, exclude_idxs_pkl=None, nper_class=2, batch_size=32, num_workers=8, img_sz=128):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nper_class = nper_class
        
        instances = []
        img_paths = sorted(glob(f'{image_dir}/*'))
        for img_path in img_paths:
            num = (img_path.split('/')[-1]).split('.')[0]
            if '_' in num:
                num = num.split('_')[-1]
            instances.append((img_path,int(num)))
        # Shift classes so minimum is 0
        if np.min([x[1] for x in instances]) != 0:
            m = np.min([x[1] for x in instances])
            instances = [(x[0],x[1]-m) for x in instances]
            
        all_labels = np.unique([x[1] for x in instances])
        split = len(all_labels)//2
                
        test_labels = []
        if exclude_idxs_pkl is not None:
            # Gather test trajectory indices for initial test labels
            with open(exclude_idxs_pkl, 'rb') as f:
                test_labels = np.unique(list(pickle.load(f)))
                test_labels = [x for x in test_labels if x in all_labels] # Clean test labels since we cleaned up the dataset
            
        test_diff = split - len(test_labels)
        rest_labels = [x for x in all_labels if x not in test_labels]
        train_labels = rest_labels[:-test_diff]
        test_labels += rest_labels[-test_diff:]
        self.train_instances = [x for x in instances if x[1] in train_labels]
        self.test_instances  = [x for x in instances if x[1] in test_labels]
        
        print(f'Train: {len(train_labels)}, Test: {len(test_labels)}')
        
        self.train_dataset = CommonDataset(self.train_instances, transform_type='all', img_sz=img_sz)
        self.test_dataset  = CommonDataset(self.test_instances,  transform_type='all', img_sz=img_sz)

        self.test_batch_size = 10 # emulates how many landmarks are detected per frame
        
        self.train_sampler = CustomMPerClassSampler(self.train_dataset.labels, m=nper_class,  batch_size=batch_size)
        # Val dataset is test dataset but 2 instances per class and train batch size
        self.val_sampler   = CustomMPerClassSampler(self.test_dataset.labels,  m=nper_class,  batch_size=batch_size)
        # Test batches have 1 unique crater per frame
        self.test_sampler  = CustomMPerClassSampler(self.test_dataset.labels,  m=1, batch_size=self.test_batch_size)

    def __len__(self):
        return len(self.train_dataset)
    
    def collate_fn(self, data):
        imgs, lbls, T128, T32, T16, T8, T4 = zip(*data)
        imgs = torch.stack(imgs)
        lbls = torch.Tensor(lbls).to(torch.int64)
        return imgs, lbls, np.array(list(T128)), np.array(list(T32)), np.array(list(T16)), np.array(list(T8)), np.array(list(T4))
    
    def make_new_test_ds_dl(self, transform_type='none', repetitions=1, sampler=False, img_sz=128):
        self.new_instances = self.test_instances*repetitions
        ds = CommonDataset(self.name, self.new_instances,  transform_type=transform_type, img_sz=img_sz)
        if sampler:
            sampler = CustomMPerClassSampler(ds.labels, m=1, batch_size=self.batch_size)
            return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler, collate_fn=self.collate_fn)
        else:
            return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=True)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=self.num_workers, collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=self.num_workers, collate_fn=self.collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, sampler=self.test_sampler, num_workers=self.num_workers, collate_fn=self.collate_fn)
