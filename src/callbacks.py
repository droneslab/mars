import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
import torch
torch.set_printoptions(sci_mode=False)
import numpy as np
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from eval import inference_matching
import umap
import umap.plot
import matplotlib.pyplot as plt
from kornia.enhance import add_weighted        
import matplotlib
import matplotlib.cm as cm
# plt.switch_backend('agg')
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import showCams, showAtts, untransform
import wandb
import torchvision as tv
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torchvision.transforms.functional as TVF
from alive_progress import alive_bar
import PIL
import io
from eval import Eval_RecallAtK, Eval_TestSetRecogAccuracy, Eval_BlenderLRORecogAccuracy, Eval_BlenderLROLostInSpace
from pytorch_grad_cam import EigenCAM
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
        
class VisEmbeddings(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        model = trainer.model.eval()
        # dataset = trainer.datamodule.val_dataset
        dataset = trainer.datamodule.val_dataloader().dataset
        n = 10 # How many clusters to plot
        embeddings = []
        ys = []
        for i in np.random.choice(np.arange(dataset.num_craters), n):
            imgs = dataset.get_all_instances(i)
            ys += [i] * imgs.shape[0]
            feats = model(imgs.cuda())
            embeddings.append(feats.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        
        print("Creating UMAP plot...")
        ys = np.array(ys)    
        # mapper = umap.UMAP(n_neighbors=15, min_dist=0.99, n_components=2, metric='euclidean').fit(embeddings)
        mapper = umap.UMAP(n_neighbors=50, min_dist=0.25).fit(embeddings)
        umap.plot.points(mapper, labels=ys, theme='fire', show_legend=False)
        ax = plt.gca()
        fig = ax.figure
        fig.tight_layout()
        canvas = ax.figure.canvas 
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_img = plot_img.reshape(canvas.get_width_height()[::-1] + (3,)).copy()
        
        trainer.logger.log_metrics({
            "UMAP Plots": [wandb.Image(plot_img, caption="UMAP Projection (@n=10)")]
        })
        # trainer.logger.experiment.add_image("umap_projection", plot_img, trainer.current_epoch+1, dataformats='HWC')
        plt.close('all')
        model.train()
        
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
# Compute attention MAE of each untransformed testing sample
# Measure of how "aligned" the attention is between them
class AttMAE(Callback):
    def on_train_end(self, trainer, pl_module):
        print('\n----- Attention Map MAEs')
        trainer.model.eval()
        
        l1 = torch.nn.L1Loss(reduction='mean')
        resnext = trainer.model.feature_extractor
        dl = trainer.datamodule.val_dataloader()        
        
        maes = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        with alive_bar(len(dl)) as bar:
            for batch in dl:
                xs,ys,T128s,T32s,T16s,T8s,T4s = batch
                idxA = torch.arange(0,xs.shape[0],2)
                idxB = torch.arange(1,xs.shape[0],2)
                feats,atts = resnext.features_and_attentions(xs.cuda()) # atts --> list of layer attention maps ([layer1, layer2, ...])
                Ts = [T32s, T16s, T8s, T4s]
                for i in range(len(atts)):
                    layer_att_maps = atts[i] # [B, N_res, C, H, W]
                    layer_att_maps_normal = untransform(layer_att_maps, Ts[i]) # Untransform each image in batch by it's respective T_normal
                    layer_mae = 0
                    for j in range(layer_att_maps_normal.shape[1]):
                        batched_att_map = layer_att_maps_normal[:,j,...] # [batch, C, H, W]
                        layer_mae += l1(batched_att_map[idxA,:], batched_att_map[idxB,:]).item()
                    maes[f'layer{i+1}'].append(layer_mae)
                bar()
        # Take mean of layer MAEs
        res_maes = {}
        for key in list(maes.keys()):
            res_maes[f'Attention Test MeanMAEs/{key}'] = np.mean(maes[key])
        trainer.logger.log_metrics(res_maes)
        
class CamVisGif(Callback):
    def draw_and_hstack_all_cams(self, imgAs_normal, imgBs_normal, cam_layer1As_normal, cam_layer1Bs_normal):
        visuals = np.zeros((imgAs_normal.shape[0], imgAs_normal.shape[2], imgAs_normal.shape[3]*2, imgAs_normal.shape[1]), dtype=np.uint8)
        for i in range(imgAs_normal.shape[0]):
            imgA = imgAs_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            camA = cam_layer1As_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            imgB = imgBs_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            camB = cam_layer1Bs_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            visA = show_cam_on_image(imgA, camA, use_rgb=True)
            visB = show_cam_on_image(imgB, camB, use_rgb=True)
            visuals[i,:] = np.hstack([visA, visB])
        return visuals
    
    def on_train_end(self, trainer, pl_module):
        print('\n----- Visualization Training Gifs: EigenCAMs')
        saved_data = trainer.model.saved_training_data
        
        imgAs = torch.stack(saved_data['images_a']) # [B, C, H, W]
        imgBs = torch.stack(saved_data['images_b'])
        T_normal_As = saved_data['T128_a'] # [compose1, compose2, ...], len() -> B
        T_normal_Bs = saved_data['T128_b']
        
        cam_layer1As = torch.stack(saved_data['cam_layer1s_a'])
        cam_layer1Bs = torch.stack(saved_data['cam_layer1s_b'])
        cam_layer2As = torch.stack(saved_data['cam_layer2s_a'])
        cam_layer2Bs = torch.stack(saved_data['cam_layer2s_b'])
        cam_layer3As = torch.stack(saved_data['cam_layer3s_a'])
        cam_layer3Bs = torch.stack(saved_data['cam_layer3s_b'])
        cam_layer4As = torch.stack(saved_data['cam_layer4s_a'])
        cam_layer4Bs = torch.stack(saved_data['cam_layer4s_b'])
        
        # Images and attention maps get untransformed to upright position
        imgAs_normal = untransform(imgAs, T_normal_As)
        imgBs_normal = untransform(imgBs, T_normal_Bs)
                
        cam_layer1As_normal = untransform(cam_layer1As, T_normal_As)
        cam_layer1Bs_normal = untransform(cam_layer1Bs, T_normal_Bs)
        cam_layer2As_normal = untransform(cam_layer2As, T_normal_As)
        cam_layer2Bs_normal = untransform(cam_layer2Bs, T_normal_Bs)     
        cam_layer3As_normal = untransform(cam_layer3As, T_normal_As)
        cam_layer3Bs_normal = untransform(cam_layer3Bs, T_normal_Bs)     
        cam_layer4As_normal = untransform(cam_layer4As, T_normal_As)
        cam_layer4Bs_normal = untransform(cam_layer4Bs, T_normal_Bs)     
        cam_layer1_vis = self.draw_and_hstack_all_cams(imgAs_normal, imgBs_normal, cam_layer1As_normal, cam_layer1Bs_normal).transpose(0,3,1,2)
        cam_layer2_vis = self.draw_and_hstack_all_cams(imgAs_normal, imgBs_normal, cam_layer2As_normal, cam_layer2Bs_normal).transpose(0,3,1,2)
        cam_layer3_vis = self.draw_and_hstack_all_cams(imgAs_normal, imgBs_normal, cam_layer3As_normal, cam_layer3Bs_normal).transpose(0,3,1,2)
        cam_layer4_vis = self.draw_and_hstack_all_cams(imgAs_normal, imgBs_normal, cam_layer4As_normal, cam_layer4Bs_normal).transpose(0,3,1,2)    
        trainer.logger.log_metrics({'EigenCAM Train Gifs/layer1': wandb.Video(cam_layer1_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'EigenCAM Train Gifs/layer2': wandb.Video(cam_layer2_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'EigenCAM Train Gifs/layer3': wandb.Video(cam_layer3_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'EigenCAM Train Gifs/layer4': wandb.Video(cam_layer4_vis, fps=1, format='gif')})
        
class AttVisGif(Callback):
    def draw_and_hstack_all_atts(self, imgAs_normal, imgBs_normal, att_layer1As_normal, att_layer1Bs_normal):
        visuals = np.zeros((imgAs_normal.shape[0], imgAs_normal.shape[2], imgAs_normal.shape[3]*2, imgAs_normal.shape[1]), dtype=np.uint8)
        for i in range(imgAs_normal.shape[0]):
            imgA = imgAs_normal[i,:]
            attA = att_layer1As_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            imgB = imgBs_normal[i,:]
            attB = att_layer1Bs_normal[i,:].permute(1,2,0).detach().cpu().numpy()
            normA = matplotlib.colors.Normalize(vmin=attA.min().item(), vmax=attA.max().item())
            normB = matplotlib.colors.Normalize(vmin=attB.min().item(), vmax=attB.max().item())
            heatA = torch.Tensor(cm.jet(normA(np.squeeze(attA)))[:,:,:3]).permute(2,0,1)
            heatB = torch.Tensor(cm.jet(normB(np.squeeze(attB)))[:,:,:3]).permute(2,0,1)
            visA = add_weighted(imgA.cuda(), 0.5, heatA.cuda(), 0.5, 0.)
            visB = add_weighted(imgB.cuda(), 0.5, heatB.cuda(), 0.5, 0.)
            vis = np.hstack([visA.permute(1,2,0).detach().cpu().numpy(), visB.permute(1,2,0).detach().cpu().numpy()])
            vis = vis*255
            visuals[i,:] = vis
        return visuals
    
    def on_train_end(self, trainer, pl_module):
        
        
        
        print('\n----- Visualization Training Gifs: Spatial Attention')
        saved_data = trainer.model.saved_training_data
        
        imgAs = torch.stack(saved_data['images_a']) # [B, C, H, W]
        imgBs = torch.stack(saved_data['images_b'])
        T128_As = saved_data['T128_a'] # [compose1, compose2, ...], len() -> B
        T128_Bs = saved_data['T128_b']
        T32_As = saved_data['T32_a']
        T32_Bs = saved_data['T32_b']
        T16_As = saved_data['T16_a']
        T16_Bs = saved_data['T16_b']
        T8_As = saved_data['T8_a']
        T8_Bs = saved_data['T8_b']
        T4_As = saved_data['T4_a']
        T4_Bs = saved_data['T4_b']
        
        # Untransform attention maps
        layer1As = untransform(torch.stack(saved_data['layer1s_a']).mean(dim=1, keepdim=True), T32_As)
        layer1Bs = untransform(torch.stack(saved_data['layer1s_b']).mean(dim=1, keepdim=True), T32_Bs)
        layer2As = untransform(torch.stack(saved_data['layer2s_a']).mean(dim=1, keepdim=True), T16_As)
        layer2Bs = untransform(torch.stack(saved_data['layer2s_b']).mean(dim=1, keepdim=True), T16_Bs)
        layer3As = untransform(torch.stack(saved_data['layer3s_a']).mean(dim=1, keepdim=True), T8_As)
        layer3Bs = untransform(torch.stack(saved_data['layer3s_b']).mean(dim=1, keepdim=True), T8_Bs)
        layer4As = untransform(torch.stack(saved_data['layer4s_a']).mean(dim=1, keepdim=True), T4_As)
        layer4Bs = untransform(torch.stack(saved_data['layer4s_b']).mean(dim=1, keepdim=True), T4_Bs)
        
        # Attention maps get resized to match image -- [B, 1, H, W]
        HW = imgAs.shape[-1]
        layer1As = TVF.resize(layer1As, HW, antialias=True)
        layer1Bs = TVF.resize(layer1Bs, HW, antialias=True)
        layer2As = TVF.resize(layer2As, HW, antialias=True)
        layer2Bs = TVF.resize(layer2Bs, HW, antialias=True)
        layer3As = TVF.resize(layer3As, HW, antialias=True)
        layer3Bs = TVF.resize(layer3Bs, HW, antialias=True)
        layer4As = TVF.resize(layer4As, HW, antialias=True)
        layer4Bs = TVF.resize(layer4Bs, HW, antialias=True)
        
        # Images and attention maps get untransformed to upright position
        imgAs_normal = untransform(imgAs, T128_As)
        imgBs_normal = untransform(imgBs, T128_Bs)
        
        att_layer1_vis = self.draw_and_hstack_all_atts(imgAs_normal, imgBs_normal, layer1As, layer1Bs).transpose(0,3,1,2)
        att_layer2_vis = self.draw_and_hstack_all_atts(imgAs_normal, imgBs_normal, layer2As, layer2Bs).transpose(0,3,1,2)
        att_layer3_vis = self.draw_and_hstack_all_atts(imgAs_normal, imgBs_normal, layer3As, layer3Bs).transpose(0,3,1,2)
        att_layer4_vis = self.draw_and_hstack_all_atts(imgAs_normal, imgBs_normal, layer4As, layer4Bs).transpose(0,3,1,2)
        trainer.logger.log_metrics({'Attention Train Gifs/layer1[-1]': wandb.Video(att_layer1_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'Attention Train Gifs/layer2[-1]': wandb.Video(att_layer2_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'Attention Train Gifs/layer3[-1]': wandb.Video(att_layer3_vis, fps=1, format='gif')})
        trainer.logger.log_metrics({'Attention Train Gifs/layer4[-1]': wandb.Video(att_layer4_vis, fps=1, format='gif')})

# Visualize heatmaps (from attention) of a batch of testing samples
class AttVis(Callback):
    def on_train_end(self, trainer, pl_module):
        print('\n----- Attention Heatmap Grid')
        trainer.model.eval()
        
        resnext = trainer.model.feature_extractor
        ds = trainer.datamodule.test_dataloader().dataset
        instances = sorted(ds.instances)
        ds = CommonDataset('vis', instances, transform_type='all')
        num_images = 64
        batch_size = 8
        sampler = CustomMPerClassSampler(ds.labels, 2, batch_size, shuffle=False)
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=trainer.datamodule.collate_fn, sampler=sampler)
        
        num_iters = 1 if batch_size >= num_images else num_images//batch_size
            
        all_heatmaps = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        all_maes = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        
        for idx,batch in enumerate(dl):
            if idx == num_iters:
                break
            xs,ys,T128s,T32s,T16s,T8s,T4s = batch
            if num_images < xs.shape[0]:
                xs = xs[:num_images]
                T128s = T128s[:num_images]
                T32s = T32s[:num_images]
                T16s = T16s[:num_images]
                T8s  = T8s[:num_images]
                T4s  = T4s[:num_images]
            
            feats,atts = resnext.features_and_attentions(xs.cuda()) 
            att_ts = [T32s, T16s, T8s, T4s]

            for i in range(4):
                name = f'layer{i+1}'
                print('\t',f'Batch {idx}: {name}')
                layer_atts = atts[i]
                layer_ts = att_ts[i]
                heatmaps, maes = showAtts(xs, layer_atts, T128s, layer_ts)
                all_heatmaps[name] += heatmaps
                maes = list(maes.detach().cpu().numpy().flatten())
                all_maes[name] += maes
            
        for lname in list(all_heatmaps.keys()):
            grid = tv.utils.make_grid(all_heatmaps[lname], nrow=8)
            grid_img = tv.transforms.functional.to_pil_image(grid)
            maes = all_maes[lname]
            mae_str = ' '.join([f'{x:.3f}' for x in maes])
            trainer.logger.log_metrics({
                f"Attention Test Heatmap Grids/{lname}": [wandb.Image(grid_img, caption=(mae_str + f' (mean: {np.mean(maes)})'))],
            })
        
class CamVis(Callback):
    def on_train_end(self, trainer, pl_module):
        print('\n----- EigenCAM Heatmap Grid')
        trainer.model.eval()
        
        resnext = trainer.model.feature_extractor
        ds = trainer.datamodule.test_dataloader().dataset
        instances = sorted(ds.instances)
        ds = CommonDataset('vis', instances, transform_type='all')
        num_images = 64
        batch_size = 8
        sampler = CustomMPerClassSampler(ds.labels, 2, batch_size, shuffle=False)
        dl = DataLoader(ds, batch_size, shuffle=False, collate_fn=trainer.datamodule.collate_fn, sampler=sampler)
        
        num_iters = 1 if batch_size >= num_images else num_images//batch_size
            
        all_heatmaps = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        all_maes = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
        for idx,batch in enumerate(dl):
            if idx == num_iters:
                break
            xs,ys,T128s,T32s,T16s,T8s,T4s = batch
            if num_images < xs.shape[0]:
                xs = xs[:num_images]
                T128s = T128s[:num_images]
            
            for target,name in zip([resnext.layer1[-1], resnext.layer2[-1], resnext.layer3[-1], resnext.layer4[-1]], ['layer1', 'layer2', 'layer3', 'layer4']):
                print('\t',f'Batch {idx}: {name}')
                heatmaps, maes = showCams(resnext, xs, T128s, target, cuda=True)
                all_heatmaps[name] += heatmaps
                maes = list(maes.numpy().flatten())
                all_maes[name] += maes
            
        for lname in list(all_heatmaps.keys()):
            grid = tv.utils.make_grid(all_heatmaps[lname], nrow=8)
            grid_img = tv.transforms.functional.to_pil_image(grid)
            maes = all_maes[lname]
            mae_str = ' '.join([f'{x:.3f}' for x in maes])
            trainer.logger.log_metrics({
                f"EigenCAM Test Heatmap Grids/{lname}": [wandb.Image(grid_img, caption=(mae_str + f' (mean: {np.mean(maes)})'))],
            })
            
# Compute EigenCAM MAE of each untransformed testing sample
class CamMAE(Callback):
    def on_train_end(self, trainer, pl_module):
        print('\n----- EigenCAM Average MAE on Test Set')
        trainer.model.eval()
        
        l1 = torch.nn.L1Loss(reduction='mean')
        resnext = trainer.model.feature_extractor
        dl = trainer.datamodule.val_dataloader()        
        
        for target,lname in zip([resnext.layer1[-1], resnext.layer2[-1], resnext.layer3[-1], resnext.layer4[-1]], ['layer1', 'layer2', 'layer3', 'layer4']):
            gc = EigenCAM(model=resnext, target_layers=[target], use_cuda=True)
            layer_mae = []
            print(f'\t{lname}')
            with alive_bar(len(dl)) as bar:
                for batch in dl:
                    xs,ys,T128s,T32s,T16s,T8s,T4s = batch
                    idxA = torch.arange(0,xs.shape[0],2)
                    idxB = torch.arange(1,xs.shape[0],2)
                    cams = gc(input_tensor=xs, aug_smooth=True, eigen_smooth=True)
                    cams = torch.tensor(cams).unsqueeze(1)
                    cams_normal = untransform(cams, T128s)
                    layer_mae.append(l1(cams_normal[idxA,...], cams_normal[idxB,...]).item())
                    bar()
            trainer.logger.log_metrics({f'EigenCAM Test MeanMAEs/{lname}': np.mean(layer_mae)})
            
class RobustSurfaceCallback(Callback):
    def normalize_image(self, x):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))

    def read_image(self, path):
        img = read_image(path, mode=ImageReadMode.RGB).float()
        return img
    
    def on_train_end(self, trainer, pl_module):
        print('\n----- Robustness Surface: Rotation vs. Translation')
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        trainer.model.eval()
        dataset = trainer.datamodule.test_dataloader().dataset
        if 'Lunar' in str(dataset) or 'AID' in str(dataset) or 'RESISC45' in str(dataset):
            path = dataset.instances[0][0]
        else:
            path = dataset.image_dir + dataset.instances[0][0]
        img = self.read_image(path)        
        transform = transforms.Compose([
            transforms.Resize((128,128), antialias=True),
            transforms.Lambda(self.normalize_image)
        ])
        img = transform(img).cuda()
        stock_emb = trainer.model(img[None,:])
        
        xs = list(range(-2,3))
        ys = list(range(-2,3))
        t_grid = np.stack(np.meshgrid(xs,ys), -1).reshape(-1, 2)
        translations = [tuple(x) for x in list(t_grid)]
        rotations = list(range(-180,190,15))
        dists = np.zeros((len(translations), len(rotations)))
                
        mean_dist = 0
        with alive_bar(len(translations)*len(rotations)) as bar:
            for i,t in enumerate(translations):
                for y,r in enumerate(rotations):
                    rot_img = transforms.functional.affine(img, r, t, 1, 0)
                    emb = trainer.model(rot_img[None,:])
                    sim = cos(stock_emb, emb)
                    dist = sim.detach().cpu().item()
                    mean_dist += dist
                    dists[i][y] = dist
                    bar()
        mean_dist = mean_dist/len(translations)*len(rotations)

        X,Y = np.meshgrid(range(len(translations)),range(len(rotations)))
        fig, ax = plt.subplots()
        cont = plt.contourf(X,Y, dists, 100, cmap='RdGy', vmin=0, vmax=1)
        plt.colorbar(cont, ax=ax)
        ax.set_xlabel('Translation')
        ax.set_ylabel('Rotation')
        
        ax.set_xticks(list(range(0,len(translations),3)))
        ax.set_yticks(list(range(0,len(rotations),3)))        
        t_lbls = [f'{translations[x][0],translations[x][1]}' for x in list(range(0,len(translations),3))]
        r_lbls = [f'{rotations[x]}{chr(176)}' for x in list(range(0,len(rotations),3))]
        ax.set_xticklabels(t_lbls)
        ax.set_yticklabels(r_lbls)
                
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        contour_img = PIL.Image.open(buf)
        trainer.logger.log_metrics({
            f"Robustness Surface": [wandb.Image(contour_img, caption=f"Mean cosine: {mean_dist}")]
        })
            
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