import sys
sys.path.append('../')
from utils import load_config, non_max_suppression_fast, show_image_gray, show_images_gray
from model import MetricLDN
import numpy as np
import torch
from alive_progress import alive_bar
import cv2
from torchvision import transforms
from matcher import FaissMatcher
import faiss
import torch.nn.functional as F
import wandb
from pathlib import Path
import argparse
import torchvision as tv
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import EigenCAM
import matplotlib.pyplot as plt

        
# Inference matching experiment over a dataset, records accuracy metrics and such
def inference_matching(dataloader, model, match_thresh=0.9, gpu=False):
    matcher = FaissMatcher(match_thresh=match_thresh, gpu=gpu)
    # Iterate through dataset (i.e. landmarks) -- "Detect landmarks in image"
    with alive_bar(len(dataloader)) as bar:
        for xs,ys,T128,T32,T16,T8,T4 in dataloader:
            xs = xs.cuda() if gpu else xs
            feats = model(xs)
            matcher.update(feats, ys)
            bar()
    return matcher

def add_detection_noise(l,r,t,b):
    # Random bounds scale (shrink)
    bscale = np.random.randint(10)
    l += bscale
    r -= bscale
    t += bscale
    b -= bscale
    # Random translation
    txy = np.random.randint(-10,11,2)
    l += txy[0]
    r += txy[0]
    t += txy[1]
    b += txy[1]
    return [b if b >= 0 else 0 for b in [l,r,t,b]]

def Eval_BlenderLRORecogAccuracy(logger, model, annofile, match_thresh=0.9, nms_thresh=0.5, gpu=True, transType='none', img_sz=128):    
    with open(annofile, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    
    detection_noise = False
    
    if transType == 'none':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'ill':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.25, 0., 0., 0.),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'trans':
        detection_noise = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'rot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(15,345)),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'all':
        detection_noise = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.25, 0., 0., 0.),
            transforms.RandomRotation(degrees=(15,345)),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
        
    # Instantiate matcher
    matcher = FaissMatcher(match_thresh=match_thresh, gpu=gpu)
    
    # Iterate over frames
    craters_in_frame = {} # ground truth crater numbers in each frame
    with alive_bar(len(lines)) as bar:
        for line in lines:
            splits = line.split(' ')
            img_path = splits[0]
            frame_num = int(img_path.split('/')[-1].split('.png')[0])
            craters_in_frame[frame_num] = []
            boxes = splits[1:] # each box is in l,r,t,b format
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get all boxes and NMS
            parsed = np.zeros((len(boxes), 5))
            for i in range(len(boxes)):
                box = boxes[i]
                b_vals = box.split(',')
                idx = b_vals[0]
                # box vals are l,r,t,b
                l,r,t,b = [int(x) for x in b_vals[1:]]
                parsed[i,0] = l
                parsed[i,1] = t
                parsed[i,2] = r
                parsed[i,3] = b
                parsed[i,4] = idx
            keeps = non_max_suppression_fast(parsed, nms_thresh)
            for box in keeps:
                l,t,r,b,idx = box
            
            # stack up a numpy array of box (detection) crops, gather ground truth crater num
            batch_detections = torch.empty((len(keeps),3,img_sz,img_sz))
            if gpu:
                batch_detections = batch_detections.cuda()
            truth_ids = torch.empty((len(keeps)), dtype=torch.int64)
            for i in range(len(keeps)):
                box = keeps[i]
                bvals = box[:-1]
                truth_idx = box[-1]
                craters_in_frame[frame_num].append(truth_idx)
                truth_ids[i] = truth_idx
                l,t,r,b = bvals
                if detection_noise:
                    l,r,t,b = add_detection_noise(l,r,t,b) # Add noise to bbox detection
                else:
                    l,r,t,b = [b if b >= 0 else 0 for b in [l,r,t,b]]
                # Crop frame at detection
                crop = img[t:b, l:r]
                cropT = transform(crop)
                batch_detections[i,:] = cropT
            
            # Produce embeddings for batch of detections, update matcher    
            batch_embs = model(batch_detections)
            matcher.update(batch_embs, truth_ids)    
            bar()
    matcher.compute_results()
    results = {f'LRO Recognition/{transType}_{k}': v for k, v in matcher.results.items()}
    logger.log_metrics(results)
    
def Eval_BlenderLROLostInSpace(logger, model, annofile, match_thresh=0.9, gpu=True, transType='none', img_sz=128):
    with open(annofile, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    
    detection_noise = False
    
    if transType == 'none':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'ill':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.25, 0., 0., 0.),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'trans':
        detection_noise = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'rot':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(15,345)),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
    elif transType == 'all':
        detection_noise = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(0.25, 0., 0., 0.),
            transforms.RandomRotation(degrees=(15,345)),
            transforms.Resize((img_sz,img_sz), antialias=True),
        ])
        
    # Instantiate matcher
    matcher = FaissMatcher(match_thresh=match_thresh, gpu=gpu)
    
    # First orbit is frames 0-703 (first 704 lines)
    first_orbit_lines = lines[:704]
    second_orbit_lines = lines[704:1407]
    third_orbit_lines = lines[1407:]
    
    # Randomly shuffle lines to demonstrate "lost in space" problem
    np.random.shuffle(second_orbit_lines)
    np.random.shuffle(third_orbit_lines)
        
    print("Seeding index with first orbit detections...")
    with alive_bar(len(first_orbit_lines)) as bar:
        for line in first_orbit_lines:
            splits = line.split(' ')
            img_path = splits[0]
            frame_num = int(img_path.split('/')[-1].split('.png')[0])
            boxes = splits[1:] # each box is in l,r,t,b format
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # stack up a numpy array of box (detection) crops, gather ground truth crater num
            batch_detections = torch.empty((len(boxes),3,img_sz,img_sz))
            if gpu:
                batch_detections = batch_detections.cuda()
            truth_ids = torch.empty((len(boxes)), dtype=torch.int64)
            for i in range(len(boxes)):
                box = boxes[i]
                b_vals = box.split(',')
                idx = b_vals[0]
                # box vals are l,r,t,b
                l,r,t,b = [int(x) for x in b_vals[1:]]
                truth_ids[i] = int(idx)
                if detection_noise:
                    l,r,t,b = add_detection_noise(l,r,t,b) # Add noise to bbox detection
                else:
                    l,r,t,b = [b if b >= 0 else 0 for b in [l,r,t,b]]
                # Crop frame at detection
                crop = img[t:b, l:r]
                cropT = transform(crop)
                batch_detections[i,:] = cropT
    
            # Produce embeddings for batch of detections, update matcher
            batch_embs = model(batch_detections)
            matcher.add(batch_embs, truth_ids)    
            bar()
    print('Lost in space second orbit...')
    second_orbit_hits = 0
    with alive_bar(len(second_orbit_lines)) as bar:
        for line in second_orbit_lines:
            splits = line.split(' ')
            img_path = splits[0]
            frame_num = int(img_path.split('/')[-1].split('.png')[0])
            boxes = splits[1:] # each box is in l,r,t,b format
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # stack up a numpy array of box (detection) crops, gather ground truth crater num
            batch_detections = torch.empty((len(boxes),3,img_sz,img_sz))
            if gpu:
                batch_detections = batch_detections.cuda()
            truth_ids = torch.empty((len(boxes)), dtype=torch.int64)
            for i in range(len(boxes)):
                box = boxes[i]
                b_vals = box.split(',')
                idx = b_vals[0]
                # box vals are l,r,t,b
                l,r,t,b = [int(x) for x in b_vals[1:]]
                truth_ids[i] = int(idx)
                if detection_noise:
                    l,r,t,b = add_detection_noise(l,r,t,b) # Add noise to bbox detection
                else:
                    l,r,t,b = [b if b >= 0 else 0 for b in [l,r,t,b]]
                # Crop frame at detection
                crop = img[t:b, l:r]
                cropT = transform(crop)
                batch_detections[i,:] = cropT
    
            # Produce embeddings for batch of detections, update matcher
            batch_embs = model(batch_detections)
            pred_ids, truth_ids = matcher.match(batch_embs, truth_ids)
            correct_matches = (pred_ids == truth_ids).sum()
            if correct_matches >= len(boxes)//2:
                second_orbit_hits += 1
            bar()
    second_orbit_acc = second_orbit_hits/len(second_orbit_lines)*100
    print(f"Second orbit lost in space accuracy: {second_orbit_acc}%")
    print('Lost in space third orbit...')
    third_orbit_hits = 0
    with alive_bar(len(third_orbit_lines)) as bar:
        for line in third_orbit_lines:
            splits = line.split(' ')
            img_path = splits[0]
            frame_num = int(img_path.split('/')[-1].split('.png')[0])
            boxes = splits[1:] # each box is in l,r,t,b format
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # stack up a numpy array of box (detection) crops, gather ground truth crater num
            batch_detections = torch.empty((len(boxes),3,img_sz,img_sz))
            if gpu:
                batch_detections = batch_detections.cuda()
            truth_ids = torch.empty((len(boxes)), dtype=torch.int64)
            for i in range(len(boxes)):
                box = boxes[i]
                b_vals = box.split(',')
                idx = b_vals[0]
                # box vals are l,r,t,b
                l,r,t,b = [int(x) for x in b_vals[1:]]
                truth_ids[i] = int(idx)
                if detection_noise:
                    l,r,t,b = add_detection_noise(l,r,t,b) # Add noise to bbox detection
                else:
                    l,r,t,b = [b if b >= 0 else 0 for b in [l,r,t,b]]
                # Crop frame at detection
                crop = img[t:b, l:r]
                cropT = transform(crop)
                batch_detections[i,:] = cropT
    
            # Produce embeddings for batch of detections, update matcher
            batch_embs = model(batch_detections)
            pred_ids, truth_ids = matcher.match(batch_embs, truth_ids)
            correct_matches = (pred_ids == truth_ids).sum()
            if correct_matches >= len(boxes)//2:
                third_orbit_hits += 1
            bar()
    third_orbit_acc = third_orbit_hits/len(third_orbit_lines)*100
    print(f"Third orbit lost in space accuracy: {third_orbit_acc}%")
    
    total_acc = (second_orbit_acc + third_orbit_acc)/2

    results = {
        f'LRO Lost In Space/{transType}_second_orbit': second_orbit_acc,
        f'LRO Lost In Space/{transType}_third_orbit': third_orbit_acc,
        f'LRO Lost In Space/{transType}_total': total_acc
    }
    
    logger.log_metrics(results)
    # logger.log(results)

# Compute recall @ k for each k
# seed_rounds ==> how many embeddings to put in the "gallery" before computing recall
def Eval_RecallAtK(logger, model, dataloader, ks, gallery_size, gpu=False):
    print(f'\n----- Gallery Size: {gallery_size}')
    ds = dataloader.dataset
    index = faiss.IndexFlatIP(512)
    y_index = {}
    if gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    print('seeding gallery...')
    idx = 0
    for _ in range(gallery_size):
        print(f'\tround {_}')
        with alive_bar(len(ds)) as bar:
            for x,y,T128,T32,T16,T8,T4 in ds:            
                emb = F.normalize(model(x.cuda()[None,:]))
                y_index[idx] = y
                index.add(emb)
                idx += 1
                bar()
    # If at least 1 of those nearest neighbors is a match, then the sample gets a score of 1. Otherwise it scores 0.
    hits = {}
    for k in ks:
        hits[f'recall_{k}'] = 0
    print('calculating top Ks...')
    with alive_bar(len(ds)) as bar:
        idx = 0
        for x,y,T128,T32,T16,T8,T4 in ds:
            emb = F.normalize(model(x.cuda()[None,:]))
            
            # Gather max K items
            D,I = index.search(emb, int(np.max(ks)))
            D = D.squeeze()
            I = I.squeeze()
            CI = [y_index[id.item()] for id in I]
            for k in ks: hits[f'recall_{k}'] += 1 if y in CI[:k] else 0
            bar()
    accs = {}
    for k in ks:
        accs[f'Recalls/Gallery#{gallery_size}_Recall@{k}'] = (hits[f'recall_{k}']/len(ds))*100
        print(f"    Recall@{k}: {accs[f'Recalls/Gallery#{gallery_size}_Recall@{k}']}")
    logger.log_metrics(accs)

# Perform navigation-style 1 at a time matching
def Eval_TestSetRecogAccuracy(logger, model, dataloader, transform_type, thresh=0.9, use_gpu=True, reps=1):
    matcher = inference_matching(dataloader, model, match_thresh=thresh, gpu=use_gpu)
    matcher.compute_results()
    results = {f'Test Recognition/{transform_type}_reps#{reps}_{k}': v for k, v in matcher.results.items()}
    logger.log_metrics(results)
