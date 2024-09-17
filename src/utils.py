import yaml
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from kornia.enhance import add_weighted        
import matplotlib
import matplotlib.cm as cm
import tensorflow as tf
import argparse
import torch
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms.functional as F
import numpy as np
try:
    progbar = pl.callbacks.progress.ProgressBar
except:
    progbar = pl.callbacks.progress.ProgressBarBase
    

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg

class CustomRichProgressBar(RichProgressBar):
    def __init__(self, theme, leave):
        super().__init__(theme=theme, leave=leave)  # don't forget this :)
        self.enable = True
        
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

def rich_progress_bar():
    theme=RichProgressBarTheme(
        description="white",
        progress_bar="green_yellow",
        progress_bar_finished="white",
        progress_bar_pulse="white",
        batch_progress="white",
        time="white",
        processing_speed="white",
        metrics="white",
    )
    return CustomRichProgressBar(theme=theme, leave=True)

class KerasProgressBar(progbar):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        pbar = tf.keras.utils.Progbar(trainer.num_training_batches)
        self.keras_bar = pbar

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        print(f'Epoch {trainer.current_epoch+1}/{trainer.max_epochs}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        metrics = self.get_metrics(trainer, pl_module)
        output = [(k,  float(metrics[k])) for k in metrics]
        self.keras_bar.update(batch_idx + 1, values=output)
    
    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        pbar = tf.keras.utils.Progbar(trainer.num_test_batches[0])
        self.keras_bar = pbar
        print('Test 1/1')
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.keras_bar.update(batch_idx + 1)

def show_image(tensor):
    plt.figure()
    plt.imshow(tensor.permute(1, 2, 0).detach().cpu())
    plt.show()
    plt.close()

def show_images(tensors):
    for i in range(tensors.shape[0]):
        plt.figure(i)
        plt.imshow(tensors[i,:].permute(1, 2, 0).detach().cpu() )
    plt.show()
    plt.close()

def show_images_gray(tensors, vmin=0, vmax=1):
    for i in range(tensors.shape[0]):
        plt.figure(i)
        plt.imshow(  tensors[i,:].permute(1, 2, 0).detach().cpu() , cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    plt.close()

def show_image_gray(tensor, vmin=0, vmax=1):
    plt.figure()
    plt.imshow(  tensor.permute(1, 2, 0).detach().cpu() , cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()
    plt.close()
    
def plotTensorImage(ax, tensor, title=''):
    ax.set_axis_off()
    ax.set_title(title)
    return ax.imshow(tensor.permute(1, 2, 0).detach().cpu() , cmap='gray', vmin=torch.min(tensor), vmax=torch.max(tensor))
    
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

# Untransform a batch of tensors
#   Each composed transform in T_normals relates to a respective image in batch
#   batch size and len(T_normals) expected to match
#   batch dim must be 0
def untransform(xs, T_normals):    
    return torch.stack([T_normals[i](xs[i,...]) for i in range(xs.shape[0])])

def showAtts(xs, atts, T128s, TAtts):
    l1 = torch.nn.L1Loss(reduction='none')
    last_maps = torch.mean(atts[:,-1,...], dim=1, keepdim=True)
    
    # Untransform atts and images
    imgs_normal = untransform(xs, T128s)
    atts_normal = untransform(last_maps, TAtts)
    atts_normal_resized = F.resize(atts_normal, [xs.shape[-1]], antialias=True)
        
    p1 = torch.arange(0,imgs_normal.shape[0],2)
    p2 = torch.arange(1,imgs_normal.shape[0],2)    
    maes = l1(atts_normal_resized[p1,...], atts_normal_resized[p2,...]).mean(dim=-2).mean(dim=-1)
    
    heatmaps = []
    for i in range(imgs_normal.shape[0]):
        img = imgs_normal[i,:]
        att = atts_normal_resized[i,:].permute(1, 2, 0).cpu().detach().numpy()
        norm = matplotlib.colors.Normalize(vmin=att.min().item(), vmax=att.max().item())
        heat = torch.Tensor(cm.jet(norm(np.squeeze(att)))[:,:,:3]).permute(2,0,1)
        vis = add_weighted(img.cuda(), 0.5, heat.cuda(), 0.5, 0.)
        heatmaps.append(vis)
        
    return heatmaps, maes

def showCams(model, xs, T_normals, target, cuda=True):
    l1 = torch.nn.L1Loss(reduction='none')
    target_layers = [target]
    gc = EigenCAM(model=model, target_layers=target_layers, use_cuda=cuda)
    cams = gc(input_tensor=xs, aug_smooth=True, eigen_smooth=True)
    cams = torch.tensor(cams).unsqueeze(1)
    
    # Untransform atts and images
    imgs_normal = untransform(xs, T_normals)
    cams_normal = untransform(cams, T_normals)
        
    p1 = torch.arange(0,cams_normal.shape[0],2)
    p2 = torch.arange(1,cams_normal.shape[0],2)
    maes = l1(cams_normal[p1,...], cams_normal[p2,...]).mean(dim=-2).mean(dim=-1)
    
    heatmaps = []
    for i in range(imgs_normal.shape[0]):
        vis = show_cam_on_image(imgs_normal[i,:].permute(1, 2, 0).cpu().detach().numpy(),
                                cams_normal[i,:].permute(1, 2, 0).cpu().detach().numpy(),
                                use_rgb=True)
        heatmaps.append(torch.from_numpy(vis).permute(2,0,1))
        
    return heatmaps, maes
