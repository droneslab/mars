from pytorch_lightning.callbacks import Callback
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
import torch
torch.set_printoptions(sci_mode=False)
import numpy as np
from eval import inference_matching
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import untransform
import torchvision as tv
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from alive_progress import alive_bar
import PIL
import io
from eval import Eval_RecallAtK, Eval_TestSetRecogAccuracy, Eval_BlenderLRORecogAccuracy, Eval_BlenderLROLostInSpace
from torch.utils.data import DataLoader
from datasets.sampler import CustomMPerClassSampler
from datasets.datamodule_factory import CommonDataset

def epochCkpt():
    return ModelCheckpoint(
                save_top_k=1,
                monitor='epoch',
                mode='max',
                filename='ckpt-{epoch}',
            )
    
def faissCkpt():
    return ModelCheckpoint(
                save_top_k=1,
                monitor="faiss_total_recognition_accuracy",
                mode="max",
                filename="ckpt-{epoch}-{faiss_total_recognition_accuracy:.4f}",
            )

class EmbeddingProjector(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        dataset = trainer.train_dataloader.dataset.datasets
        perm = torch.randperm(dataset.max_label())
        num_imgs = dataset.get_all_instances(0).shape[0]
        ex,_ = dataset[0]
        channels = ex.shape[0]
        img_size = ex.shape[1]
        n = 10 # How many clusters to plot
        embeddings = []
        ys = []
        all_imgs = torch.zeros((n*num_imgs,channels,img_size,img_size))
        cnt = 0
        for i in perm[:n]:
            imgs = dataset.get_all_instances(i)
            all_imgs[cnt:cnt+num_imgs,:] = imgs
            ys += [i.numpy()] * num_imgs
            feats = trainer.model(imgs.cuda())
            embeddings.append(feats.detach().cpu().numpy())
            cnt += num_imgs
        embeddings = np.concatenate(embeddings, axis=0)
        trainer.logger.experiment.add_embedding(embeddings, metadata=ys, label_img=all_imgs, global_step=pl_module.current_epoch+1)

class LogAccuracies(Callback):
    # Collate function for hirise-style datasets with transforms
    def collate_fn(self, data):
        imgs, lbls, T128, T32, T16, T8, T4 = zip(*data)
        imgs = torch.stack(imgs)
        lbls = torch.Tensor(lbls).to(torch.int64)
        return imgs, lbls
    
    ### convenient function from pytorch-metric-learning ###
    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester()
        return tester.get_all_embeddings(dataset, model, collate_fn=self.collate_fn)

    ### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
    def get_accuracies(self, dataloader, model, accuracy_calculator):
        embeddings, labels = self.get_all_embeddings(dataloader.dataset, model)
        labels = labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(embeddings, labels)
        return accuracies
    
    def on_train_end(self, trainer, pl_module):
        print('\n----- NMI/AMI Calc')
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1", 'NMI', 'AMI'), k=1)
        model = trainer.model.eval()
        dl = trainer.datamodule.test_dataloader()
        np.random.shuffle(dl.dataset.instances)
        accuracies = self.get_accuracies(dl, model, accuracy_calculator)
        for key,val in accuracies.items():
            accuracies[key] = val*100
        trainer.logger.log_metrics(accuracies)
        
class FaissInference(Callback):
    def __init__(self, match_thresh=0.99, use_gpu=False):
        super().__init__()
        self.match_thresh = match_thresh
        self.use_gpu = use_gpu
        
    def on_train_epoch_end(self, trainer, pl_module):
        print("Faiss inference matching...")
        model = trainer.model.eval()
        # dataset = trainer.datamodule.val_dataset
        dl = trainer.datamodule.val_dataloader()
        matcher = inference_matching(dl, model, match_thresh=self.match_thresh, gpu=self.use_gpu)
        matcher.compute_results()
        
        self.log('faiss_feats_in_index', float(matcher.results['num_feats']))
        self.log('faiss_mean_cosine_similarity', float(matcher.results['mean_dist']))
        self.log('faiss_correct_matches', float(matcher.results['corr_matches']))
        self.log('faiss_incorrect_matches', float(matcher.results['incorr_matches']))
        self.log('faiss_missed_matches', float(matcher.results['missed_matches']))
        self.log('faiss_index_matching_accuracy', matcher.results['index_matching_acc'])
        self.log('faiss_total_recognition_accuracy', matcher.results['total_recog_acc'])
        model.train()
        
'''
==========================================
EVAL CALLBACKS
==========================================
'''         
class Eval_RecallAtKCallback(Callback):
    def __init__(self, ks, num_galleries, gpu=False):
        super().__init__()
        self.ks = ks
        self.gpu = gpu
        self.num_galleries = num_galleries
        
    def on_train_end(self, trainer, pl_module):
        print(f"\n----- Recall@Ks")
        print(f"  Ks:            {self.ks}")
        print(f"  Gallery sizes: {self.num_galleries}")
        logger = trainer.logger
        model = trainer.model.cuda().eval()
        dl = trainer.datamodule.test_dataloader()
        for gallery_size in self.num_galleries:
            Eval_RecallAtK(logger, model, dl, self.ks, gallery_size, self.gpu)


class Eval_TestSetRecogCallback(Callback):
    def __init__(self, datamodule, transform_type, repetitions, thresh=0.9, use_gpu=True, img_sz=128):
        super().__init__()
        self.thresh = thresh
        self.use_gpu = use_gpu
        self.transform_type = transform_type
        self.dm = datamodule
        self.repetitions = repetitions
        self.img_sz = img_sz
        
    def on_train_end(self, trainer, pl_module):
        print(f"\n----- Test Set: Recognition Accuracy ({self.transform_type})")
        print(f"  thresh:      {self.thresh}")
        print(f"  use_gpu:     {self.use_gpu}")
        print(f"  repetitions: {self.repetitions}")
        logger = trainer.logger
        if self.use_gpu:
            model = trainer.model.cuda().eval()
        else:
            model = trainer.model.cpu().eval()
        
        for rep in self.repetitions:
            print(f'--- rep {rep}')
            dl = self.dm.make_new_test_ds_dl(transform_type=self.transform_type, repetitions=rep, sampler=False, img_sz=self.img_sz)
            Eval_TestSetRecogAccuracy(logger,model,dl,self.transform_type,self.thresh,self.use_gpu, reps=rep)
        
class Eval_BlenderRecogCallback(Callback):
    def __init__(self, anno_file, transType, match_thresh=0.9, nms_thresh=0.5, use_gpu=True, img_sz=128):
        super().__init__()
        self.anno_file = anno_file
        self.transType = transType
        self.match_thresh = match_thresh
        self.nms_thresh = nms_thresh
        self.use_gpu = use_gpu
        self.img_sz = img_sz
        
    def on_train_end(self, trainer, pl_module):
        print(f"\n----- Blender LRO Trajectory: Recognition Accuracy ({self.transType})")
        print(f"  anno_file:        {self.anno_file}")
        print(f"  match_thresh:     {self.match_thresh}")
        print(f"  nms_thresh:       {self.nms_thresh}")
        print(f"  use_gpu:          {self.use_gpu}")
        logger = trainer.logger
        if self.use_gpu:
            model = trainer.model.cuda().eval()
        else:
            model = trainer.model.cpu().eval()
            
        Eval_BlenderLRORecogAccuracy(logger,model,self.anno_file, self.match_thresh, self.nms_thresh, self.use_gpu, self.transType, img_sz=self.img_sz)

class Eval_BlenderLISCallback(Callback):
    def __init__(self, anno_file, transType, match_thresh=0.9, use_gpu=True, img_sz=128):
        super().__init__()
        self.anno_file = anno_file
        self.transType = transType
        self.match_thresh = match_thresh
        self.use_gpu = use_gpu
        self.img_sz = img_sz
        
    def on_train_end(self, trainer, pl_module):
        print(f"\n----- Blender LRO Trajectory: Lost In Space Accuracy")
        print(f"  anno_file:        {self.anno_file}")
        print(f"  transType:        {self.transType}")
        print(f"  match_thresh:     {self.match_thresh}")
        print(f"  use_gpu:          {self.use_gpu}")
        logger = trainer.logger
        if self.use_gpu:
            model = trainer.model.cuda().eval()
        else:
            model = trainer.model.cpu().eval()
            
        Eval_BlenderLROLostInSpace(logger, model, self.anno_file, self.match_thresh, self.use_gpu, self.transType, img_sz=self.img_sz)