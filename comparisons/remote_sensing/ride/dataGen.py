
import glob
from collections import defaultdict
import os
import numpy as np
import random
import pickle
from PIL import Image
from skimage import io
import copy
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def default_loader(path):
    return Image.open(path).convert('RGB')


def eurosat_loader(path):
    return io.imread(path)

def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return np.eye(n_classes)[x]


class DataGenSNCA_RI:

    def __init__(self, data, dataset, dataset_RI, angles=[0, 90, 180, 270], rate=0.7, imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.data_RI_dir = os.path.join(data, dataset_RI)
        # self.tvt_pth = os.path.join(data, dataset+'_tvt.pkl')
        self.rate = rate
        # with open(self.tvt_pth, 'rb') as f:
        #     self.tvt_data = pickle.load(f)
        self.angles = angles
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict() 
        # self.instance2Label = defaultdict() # each group of rotated images 

        self.imgTransform = imgTransform
        # self.imgExt = imgExt
        self.phase = phase

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        # number of images
        self.train_count = 0
        self.test_count = 0
        self.val_count = 0
        # each group of rotated images as instance
        train_ins_count = 0
        test_ins_count = 0
        
        instances = []
        img_paths = glob.glob(self.data_RI_dir + '/*.png')
        for img_path in img_paths:
            num = (img_path.split('/')[-1]).split('.png')[0]
            if '_' in num:
                num = num.split('_')[0]
            instances.append((img_path,int(num)))
        all_labels = np.unique([x[1] for x in instances])
        # Gather test trajectory indices, and form train/val/test datasets
        with open(f'../../../data/traj_test_indices/traj1.pk', 'rb') as f:
            test_labels = np.unique(list(pickle.load(f)))
        self.test_labels = [x for x in test_labels if x in all_labels] # Clean test labels since we cleaned up the dataset
        train_instances = [x for x in instances if x[1] not in self.test_labels]
        # 10% Validation split based on label
        labels = sorted(np.unique([x[1] for x in train_instances]))
        self.sceneList = np.arange(len(labels) + len(np.unique(self.test_labels)))
        split = int(len(labels) * 0.1)
        self.train_labels = labels[:-split]
        self.val_labels = labels[-split:]
        # Face losses require congruently increasing label numbers
        # train_true_labels = np.arange(len(train_labels))
        # train_label_map = dict(zip(train_labels, train_true_labels)) 
        # self.train_instances = [(x[0], train_label_map[x[1]]) for x in train_instances if x[1] in train_labels]
        self.train_instances = [x for x in train_instances if x[1] in self.train_labels]
        self.val_instances = [x for x in train_instances if x[1] in self.val_labels]
        self.test_instances = [x for x in instances if x[1] in self.test_labels]
        
        for label in self.train_labels:
            for angle in ['0', 'hf', 'vf', 'ill', 'r90', 'r180', 'r270']:
                if angle == '0':
                    imgPth = os.path.join(self.data_RI_dir, str(label)+'.png')
                else:
                    imgPth = os.path.join(self.data_RI_dir, str(label)+f'_{angle}.png')
                self.train_idx2fileDict[self.train_count] = (imgPth, label, train_ins_count)
                self.train_count += 1
            train_ins_count += 1
            
        for label in self.val_labels:
            for angle in ['0', 'hf', 'vf', 'ill', 'r90', 'r180', 'r270']:
                if angle == '0':
                    imgPth = os.path.join(self.data_RI_dir, str(label)+'.png')
                else:
                    imgPth = os.path.join(self.data_RI_dir, str(label)+f'_{angle}.png')
                self.val_idx2fileDict[self.val_count] = (imgPth, label)
                self.val_count += 1
                
        for label in self.test_labels:
            for angle in ['0', 'hf', 'vf', 'ill', 'r90', 'r180', 'r270']:
                if angle == '0':
                    imgPth = os.path.join(self.data_RI_dir, str(label)+'.png')
                else:
                    imgPth = os.path.join(self.data_RI_dir, str(label)+f'_{angle}.png')
                self.test_idx2fileDict[self.test_count] = (imgPth, label, test_ins_count)
                self.test_count += 1
            test_ins_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print(f"number of train instance: {train_ins_count}")
        print(f"number of test instance: {test_ins_count}")
        print("total number of train images: {}".format(self.train_count))
        print("total number of val images: {}".format(self.val_count))
        print("total number of test images: {}".format(self.test_count))


    def __getitem__(self, index):

        return self.__data_generation(index)

    def __data_generation(self, idx):

        if self.phase == 'train':
            imgPth, imgLb, imgInsLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb, imgInsLb = self.test_idx2fileDict[idx]

        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)

        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        if self.phase == 'val':
            return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        else:
            return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32), 'insLabel':imgInsLb}

    def __len__(self):

        if self.phase == 'train':
            return self.train_count
        elif self.phase == 'val':
            return self.val_count
        else:
            return self.test_count


flatten = lambda l: [item for sublist in l for item in sublist]

class DataGenSAP_RI(Dataset):
    """ 
    PK sampler (P categories, K samples/category) to construct mini-batches.
    follows: Cross-Batch Memory for Embedding Learning
    Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval
    https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/datasets.py
    """
    def __init__(self, data, dataset, dataset_RI, batch_size=128, samples_per_cls=4, rate=0.7, imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.data_RI_dir = os.path.join(data, dataset_RI)
        self.tvt_pth = os.path.join(data, dataset+'_tvt.pkl')
        self.rate = rate
        with open(self.tvt_pth, 'rb') as f:
            self.tvt_data = pickle.load(f)

        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.phase = phase
        self.batch_size = batch_size
        self.samples_per_cls = samples_per_cls
        # self.imgExt = imgExt
        self.imgTransform = imgTransform
        self.scene2Label = defaultdict()

        self.train_cls2fileDict = defaultdict()

        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()
        
        self.CreateIdx2fileDict()
        self.reshuffle()

    def CreateIdx2fileDict(self):

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0
        
        test_count = 0
        val_count = 0
        train_count = 0

        train_ins_count = 0
        test_ins_count = 0

        for label, scenePth in enumerate(self.sceneList):
            sceneName = os.path.basename(scenePth)
            self.scene2Label[sceneName] = label

            # subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            # random.seed(42)
            # random.shuffle(subdirImgPth)

            # train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]
            
            train_ImgNames = self.tvt_data[sceneName][self.rate]['train']
            val_ImgNames = self.tvt_data[sceneName][self.rate]['val']
            test_ImgNames = self.tvt_data[sceneName][self.rate]['test']

            self.train_numImgs += len(train_ImgNames)
            self.test_numImgs += len(test_ImgNames)
            self.val_numImgs += len(val_ImgNames)

            for imgName in train_ImgNames:
                self.train_cls2fileDict[train_ins_count] = []
                # self.instance2Label[imgName] = ins_label
                # print(imgName)
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName)
                    elif angle == 90:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_90.jpg')
                    elif angle == 180:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_180.jpg')
                    else:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_270.jpg')
                    
                    self.train_cls2fileDict[train_ins_count].append((imgPth, label, train_count, train_ins_count))
                    train_count += 1
                train_ins_count += 1

            for imgName in val_ImgNames:
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName)
                    elif angle == 90:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_90.jpg')
                    elif angle == 180:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_180.jpg')
                    else:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_270.jpg')

                    self.val_idx2fileDict[val_count] = (imgPth, label)
                    val_count += 1

            for imgName in test_ImgNames:
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName)
                    elif angle == 90:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_90.jpg')
                    elif angle == 180:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_180.jpg')
                    else:
                        imgPth = os.path.join(self.data_RI_dir, sceneName, imgName.split('.')[0]+'_270.jpg')

                    self.test_idx2fileDict[test_count] = (imgPth, label, test_ins_count)
                    test_count += 1
                test_ins_count += 1

            # for imgPth in train_subdirImgPth:
            #     # train_count is for identifying the idx of train img
            #     self.train_cls2fileDict[label].append((imgPth, label, train_count)) 
            #     train_count += 1

            # for imgPth in test_subdirImgPth:
            #     self.test_idx2fileDict[test_count] = (imgPth, label)
            #     test_count += 1
            
            # for imgPth in val_subdirImgPth:
            #     self.val_idx2fileDict[val_count] = (imgPth, label)
            #     val_count += 1
        
        # print("total number of classes: {}".format(len(self.sceneList)))
        # print("total number of train images: {}".format(self.train_numImgs))
        # print("total number of val images: {}".format(self.val_numImgs))
        # print("total number of test images: {}".format(self.test_numImgs))

        print("total number of classes: {}".format(len(self.sceneList)))
        print(f"number of train instance: {train_ins_count}")
        print(f"number of test instance: {test_ins_count}")
        print("total number of train images: {}".format(train_count))
        print("total number of val images: {}".format(val_count))
        print("total number of test images: {}".format(test_count))

    def reshuffle(self):
        image_dict = copy.deepcopy(self.train_cls2fileDict)

        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0

        # generate batches for one training epoch
        while finished == 0:
            # print(len(total_batches))
            for cls_ in classes:
                if (len(image_dict[cls_]) >= self.samples_per_cls) and (len(batch) < self.batch_size//self.samples_per_cls):
                    batch.append(image_dict[cls_][:self.samples_per_cls])
                    image_dict[cls_] = image_dict[cls_][self.samples_per_cls:]

            if len(batch) == self.batch_size//self.samples_per_cls:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        random.shuffle(total_batches)
        self.train_dataset = flatten(flatten(total_batches))

    def __getitem__(self, index):
        
        return self.__data_generation(index)

    def __data_generation(self, idx):

        if self.phase == 'train':
            imgPth, clsLb, index, imgLb = self.train_dataset[idx]
        elif self.phase == 'val':
            imgPth, clsLb = self.val_idx2fileDict[idx]
        else:
            imgPth, clsLb, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)

        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))
        if self.phase == 'train':
            return {'img': img, 'label': clsLb, 'idx':index, 'insLabel':imgLb}
        else:
            return {'img': img, 'label': clsLb}
    
    def __len__(self):
        
        if self.phase == 'train':
            return len(self.train_dataset)
        elif self.phase == 'val':
            return self.val_numImgs
        else:
            return self.test_numImgs

class DataGeneratorSplitting:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, rate=0.7, imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()

        self.tvt_pth = os.path.join(data, dataset+'_tvt.pkl')
        self.rate = rate
        with open(self.tvt_pth, 'rb') as f:
            self.tvt_data = pickle.load(f)

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        # self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            sceneName = os.path.basename(scenePth)
            self.scene2Label[sceneName] = label

            # subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            # random.seed(42)
            # random.shuffle(subdirImgPth)

            # train_subdirImgPth = subdirImgPth[:int(0.2*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.2*len(subdirImgPth)):int(0.3*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):]
            
            # train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]

            train_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['train']))
            val_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['val']))
            test_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['test']))

            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        elif self.phase == 'val':
            idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        elif self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)
        oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb, 'idx':idx, 'onehot':oneHotVec.astype(np.float32)}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)



class DataGeneratorTiplet:

    def __init__(self, data, dataset, rate=0.7, imgTransform=None, phase='train'):

        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()

        self.tvt_pth = os.path.join(data, dataset+'_tvt.pkl')
        self.rate = rate
        with open(self.tvt_pth, 'rb') as f:
            self.tvt_data = pickle.load(f)

        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.train_label2idx = defaultdict()
        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        # self.imgExt = imgExt
        self.phase = phase
        self.labels_list = None

        self.CreateIdx2fileDict()

    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        self.val_numImgs = 0

        train_count = 0
        test_count = 0
        val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            sceneName = os.path.basename(scenePth)
            self.scene2Label[sceneName] = label

            # subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            # random.seed(42)
            # random.shuffle(subdirImgPth)
            
            # train_subdirImgPth = subdirImgPth[:int(0.7*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            # test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]
            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)

            train_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['train']))
            val_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['val']))
            test_subdirImgPth = list(map(lambda x: os.path.join(scenePth, x), self.tvt_data[sceneName][self.rate]['test']))

            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)

                if label in self.train_label2idx:
                    self.train_label2idx[label].append(train_count)
                else:
                    self.train_label2idx[label] = [train_count]
                train_count += 1
                
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            for imgPth in val_subdirImgPth:
                self.val_idx2fileDict[val_count] = (imgPth, label)
                val_count += 1
        
        self.labels_list = list(range(len(self.sceneList)))

        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        self.valDataIndex = list(range(self.val_numImgs))


    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
            _, imgLb = self.train_idx2fileDict[idx]
            positive_index = idx
            while positive_index == idx:
                positive_index = np.random.choice(self.train_label2idx[imgLb])
            
            negative_label = np.random.choice(list(set(self.labels_list) - set([imgLb])))
            negative_index = np.random.choice(self.train_label2idx[negative_label])
            return self.__data_generation_triplet(idx, positive_index, negative_index)

        elif self.phase == 'val':
            idx = self.valDataIndex[index]
            return self.__data_generation(idx)
        else:
            idx = self.testDataIndex[index]
            return self.__data_generation(idx)


    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]
        if self.phase == 'val':
            imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)

        # return {'img': img, 'label': imgLb, 'idx':idx}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        return {'img': img, 'label': imgLb}

    def __data_generation_triplet(self, idx, pos_idx, neg_idx):

        anc_imgPth, anc_label = self.train_idx2fileDict[idx]
        pos_imgPth, _ = self.train_idx2fileDict[pos_idx]
        neg_imgPth, _ = self.train_idx2fileDict[neg_idx]

        anc_img = default_loader(anc_imgPth)
        pos_img = default_loader(pos_imgPth)
        neg_img = default_loader(neg_imgPth)

        if self.imgTransform is not None:
            anc_img = self.imgTransform(anc_img)
            pos_img = self.imgTransform(pos_img)
            neg_img = self.imgTransform(neg_img)

        return {'anc':anc_img, 'pos':pos_img, 'neg':neg_img, 'anc_label':anc_label}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        elif self.phase == 'val':
            return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)

class Rotation:
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x):
        x = TF.rotate(x, self.angle)
        return x


if __name__ == "__main__":
    
    # data_AID = DataGenSNCA_RI(
    #                 data='D:\data\scene', 
    #                 dataset='NWPU-RESISC45', 
    #                 dataset_RI='NWPU-RESISC45_RI'
    #     )
    from tqdm import tqdm
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    rot_transform = defaultdict()

    rot_angles = list(range(30,360,30))

    for angle in rot_angles:
        rot_transform[angle] = transforms.Compose([
                                                transforms.Resize((256,256), antialias=True),
                                                Rotation(angle),
                                                transforms.ToTensor(),
                                                normalize
                                                ])

    for rot_angle in rot_angles:
            
        test_dataGen_rot = DataGeneratorSplitting(data=r"D:\data\scene", 
                                    dataset='AID',
                                    rate=0.7,
                                    imgTransform=rot_transform[angle],
                                    phase='test')
        test_data_loader_rot = DataLoader(test_dataGen_rot, batch_size=16, num_workers=4, shuffle=False, pin_memory=True)

        # data = test_dataGen_rot[0]

        for data in tqdm(test_data_loader_rot, desc="extracting testing rot_angle"):
            imgs = data['img'].to(torch.device("cuda"))
            idxs = data['idx'].to(torch.device("cpu"))



