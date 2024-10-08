import yaml
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
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
