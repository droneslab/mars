from osgeo import gdal, osr
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import csv
import spiceypy as sp
import glob
import pandas as pd
from astropy.coordinates import SkyCoord
import argparse
from matplotlib.patches import Rectangle
import time
import sys
sys.path.append('../src/')
from model import ResNext
from utils import show_image_gray
import faiss
import gtsam
from torchvision import transforms
import torch
import cv2

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--csv', action='store', help="CSV Pose File")
# parser.add_argument('-l', '--lowres', action='store_true', help="use lowres GeoTIFF")
# args = parser.parse_args()

if 'lowres' in sys.argv:
    ds = gdal.Open('/home/tj/data/lunar_models/mosaics/WAC_GLOBAL_E000N0000_004P.tiff')
else:
    ds = gdal.Open('/home/tj/data/lunar_models/mosaics/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')
    

''' NOTATION 
crs        ==> GeoTiff CRS (PROJCRS["SimpleCylindrical MOON"])
crsGeo     ==> Moon 2000 CRS
x,y        ==> pixel coordinates
posX,posY  ==> crs coordinates
lat,long   ==> crsGeo coordinates

GeoTiff info given by: 'gdalinfo /home/tj/data/lunar_models/mosaics/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif'
'''
xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
# get CRS from dataset 
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
# Proj4 format of 'Moon 2000': https://spatialreference.org/ref/iau2000/30100/ (ALSO FROM QGIS)
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +R=1737400 +no_defs')

# Pixels --> CRS
def pxToCrs(x,y):
    # Get coordinate of pixel in PROJCRS space
    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset
    # shift to the center of the pixel
    # posX += px_w / 2.0
    # posY += px_h / 2.0
    return posX,posY

# Pixels --> CRS --> Moon 2000
def pxToLatLong(x, y):
    # First convert pixel to crs space
    posX,posY = pxToCrs(x,y)
    # Transform to Moon 2000
    t = osr.CoordinateTransformation(crs, crsGeo)
    (lon, lat, z) = t.TransformPoint(posX, posY)
    return lat,lon

# Moon 2000 --> CRS --> pixels
def latLongToPx(lat,long):
    t = osr.CoordinateTransformation(crsGeo, crs)
    (posX,posY, z) = t.TransformPoint(long, lat)
    x = int( (posX - xoffset) / px_w )
    y = int( (posY - yoffset) / px_h )
    return x,y

def visualizeTrajectory(img, lonlats, truth):
    plt.imshow(img, cmap='gray', vmin=np.min(img), vmax=np.max(img))
    i = 0
    for lonlat in lonlats:
        lon = lonlat[0]
        lat = lonlat[1]
        x,y = latLongToPx(lat,lon)
        # plt.text(x,y, str(i), color="white", fontsize=12)
        plt.plot(x,y,'co--')
        i += 1
    for lonlat in truth[:len(lonlats)]:
        lon = lonlat[0]
        lat = lonlat[1]
        x,y = latLongToPx(lat,lon)
        plt.plot(x,y,'ro--')
    plt.show()
    plt.close()
    
def visualizeCraters(img, lonlatdiams):
    for line in lonlatdiams:
        long = line[0]
        lat = line[1]
        diam_km = line[2]
        diam_m = diam_km * 1000
        diam_px = diam_m/100
        
        print(line, diam_px)

        x,y = latLongToPx(lat,long)

        fsize = int(diam_px)
        frame = img[y-fsize:y+fsize, x-fsize:x+fsize]
        
        plt.imshow(frame, cmap='gray', vmin=np.min(img), vmax=np.max(img))
        plt.show()
        plt.close()

def visualizeXYZ(poses, gt):
    fig = plt.figure()    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(poses[:,0], poses[:,1], poses[:,2])
    ax2.scatter(gt[:,0], gt[:,1], gt[:,2])
    plt.show()

def latLongToXYZ(lat,long):
    t = SkyCoord(long,lat, unit='degree', representation_type='spherical')
    return (t.cartesian.x,t.cartesian.y,t.cartesian.z)

def grabSpiceLonLats(start, end, freq):
    kernel_list = glob.glob('../data/spice_kernels/**/*')
    for filename in kernel_list:
        if 'Identifier' in filename or 'lbl' in filename:
            continue
        sp.furnsh(filename)
    
    epochs = list(pd.date_range(start=start, end=end, freq=freq))
    epochs = [x.to_pydatetime() for x in epochs]
    epochs = [x.strftime("%Y-%b-%d %H:%M:%S.%f") for x in epochs]
    
    lon_lats = []
    for epoch in epochs:
        et = sp.str2et(epoch)
        lro_pos,_ = sp.spkpos('LRO', et, 'MOON_PA', 'LT+S', 'MOON')
        t = SkyCoord(x=lro_pos[0], y=lro_pos[1], z=lro_pos[2], representation_type='cartesian')
        # Convert lon to [-180,180] (http://vikas-ke-funde.blogspot.com/2010/06/convert-longitude-0-360-to-180-to-180.html)
        lon = ((t.spherical.lon.degree+180)%360)-180
        lat = t.spherical.lat.degree
        lon_lats.append((lon,lat))
        
    return lon_lats

def grabCraterLonLats(csv_file, lonLine=0, latLine=1, diamLine=2):
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        l = 0
        lonlatdiams = []
        for line in csv_reader:
            if l == 0:
                l += 1
                continue
            long = float(line[lonLine])
            lat = float(line[latLine])
            diam_km = float(line[diamLine])
            lonlatdiams.append( (long, lat, diam_km) )
    return lonlatdiams

def convertCraterDatabasePixels(lonlatdiams):
    craters_px = []
    for line in lonlatdiams:
        lon = line[0]
        lat = line[1]
        diam_km = line[2]
        diam_m = diam_km * 1000
        diam_px = diam_m/100
        # diam_px = diam_px - int(diam_px*0.5)    
        x,y = latLongToPx(lat,lon)
        craters_px.append( (x,y,diam_px) )
    return craters_px

def checkCraterFits(xmin,xmax,ymin,ymax, crater, margin):
    x = crater[0]
    y = crater[1]
    diam_px = crater[2]
    step = diam_px/2
    if x-step-margin >= xmin and x+step+margin <= xmax and y-step-margin >= ymin and y+step+margin <= ymax:
        return True
    else:
        return False
    
# Normalize to [0,1]
def normalize_image(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128,128), antialias=True),
        transforms.Lambda(normalize_image)
    ])        
    image = transform(image)
    return image.repeat(3, 1, 1)

def augment_image(image, type):
    import torchvision.transforms.functional as TF
    class BrightChange:
        def __init__(self, bfactor):
            self.bfactor = bfactor
        def __call__(self, x):
            return TF.adjust_brightness(x, self.bfactor)
        
    if type == 'rot':
        aug = transforms.RandomRotation(degrees=(15,345))
    elif type == 'none':
        aug = transforms.Lambda(lambda x: x)
    elif type == 'bup':
        aug = BrightChange(1.30)
    elif type == 'bdown':
        aug = BrightChange(0.70)
    
    transform = transforms.Compose([
        transforms.ToTensor(), # ToTensor() will normalize the image if it has a channel
        transforms.Resize((128,128), antialias=True),
        aug,
    ])
    image = transform(image)
    return image.repeat(3, 1, 1)
    
def parseModelCkpts(ckpt_path='../src/ccr_logs/30epochs/**/**/**/checkpoints/', type='faiss'):
    # Create dictionary of model types and paths to their checkpoints
    model_ckpts = glob.glob(ckpt_path + f'*{type}*', recursive=True)
    # model_names = [m.split('SEResNeXt_')[-1].split('Loss')[0] for m in model_ckpts if 'SENet' in m]
    model_names = [m.split('ResNeXt_')[-1] for m in model_ckpts]
    model_names = [m.split('Loss')[0] if 'Loss' in m else m.split('_')[1] for m in model_names]
    model_names += [m.split('_sampler')[0].split('/')[-1] for m in model_ckpts if 'RiDe' in m]
    model_paths = dict.fromkeys(model_names)
    for model_name in list(model_paths.keys()):
        model_paths[model_name] = {}
        model_paths[model_name]['path'] = [m for m in model_ckpts if model_name in m][0]
        model_paths[model_name]['att_type'] = 'se' if 'se' in model_paths[model_name]['path'] else 'cbam'
        if 'RiDe' not in model_name:
            # model_paths[model_name]['miner'] = 'multisim' if 'MultiSimilarityMiner' in model_paths[model_name]['path'] else 'none'
            model_paths[model_name]['miner'] = 'multisim' if ('multisim' in model_paths[model_name]['path'] or 'MultiSimilarityMiner' in model_paths[model_name]['path']) else 'none'
    return model_paths

def drawVisibleCraters(ax, vis_craters, vis_crater_idxs, xmin, ymin):
    for i in range(len(vis_craters)):
        crater = vis_craters[i]
        x = crater[0]
        y = crater[1]
        diam_px = crater[2]        
        x = x-xmin
        y = y-ymin
        ax.plot(x,y,'co--', markersize=3)
        circle = plt.Circle((x,y), diam_px/2, color='c', fill=False, clip_on=False, alpha=0.5)
        ax.add_patch(circle)
        ax.text(x,y, vis_crater_idxs[i], fontsize=10, color='c')
    return ax

def write_posefile(poses, filename):
    ids = list(poses.keys())
    pose_lines = ''
    for i in ids:
        R = poses[i]['R']
        t = poses[i]['t']
        pose_lines += '{} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
            R[0,0], R[0,1], R[0,2], t[0,0], R[1,0], R[1,1], R[1,2], t[1,0], R[2,0], R[2,1], R[2,2], t[2,0]
        )
    with open(filename, 'w') as f:
        f.write(pose_lines)

def calcPose(img, fsize, craters_px, margin, last_pos, curr_pos, correspondance, num_kps=10, drawMatches=True):
    # last_pos, curr_pos ==> (lon,lat)
    # num_kps ==> patch size of num_kps x num_kps to extract around each crater center
    
    # Last frame info
    last_x,last_y = latLongToPx(last_pos[1], last_pos[0])
    last_frame = img[last_y-fsize:last_y+fsize, last_x-fsize:last_x+fsize]
    last_xmin = last_x-fsize
    last_xmax = last_x+fsize
    last_ymin = last_y-fsize
    last_ymax = last_y+fsize
    last_vis_craters = [c for c in craters_px if checkCraterFits(last_xmin,last_xmax,last_ymin,last_ymax,c,margin)]
    last_vis_crater_idxs = [craters_px.index(c) for c in last_vis_craters] # ground truth crater index number
    
    # Curr frame info
    curr_x,curr_y = latLongToPx(curr_pos[1], curr_pos[0])
    curr_frame = img[curr_y-fsize:curr_y+fsize, curr_x-fsize:curr_x+fsize]
    curr_xmin = curr_x-fsize
    curr_xmax = curr_x+fsize
    curr_ymin = curr_y-fsize
    curr_ymax = curr_y+fsize
    curr_vis_craters = [c for c in craters_px if checkCraterFits(curr_xmin,curr_xmax,curr_ymin,curr_ymax,c,margin)]
    curr_vis_crater_idxs = [craters_px.index(c) for c in curr_vis_craters] # ground truth crater index number
    
    if drawMatches:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(last_frame, cmap='gray', vmin=np.min(img), vmax=np.max(img))
        drawVisibleCraters(ax1, last_vis_craters, last_vis_crater_idxs, last_xmin, last_ymin)
        ax2.imshow(curr_frame, cmap='gray', vmin=np.min(img), vmax=np.max(img))
        drawVisibleCraters(ax2, curr_vis_craters, curr_vis_crater_idxs, curr_xmin, curr_ymin)
                        
    # Gather pts1, pts2
    pts1 = []
    pts2 = []
    for query_id,matched_id in correspondance:
        query_crater = craters_px[query_id] # Current frame crater
        matched_crater = craters_px[matched_id] # Last frame matched crater
        if matched_crater in last_vis_craters and query_crater in curr_vis_craters:
            last_x = matched_crater[0]-last_xmin
            last_y = matched_crater[1]-last_ymin
            cpatch_y = np.arange(int(last_y-(num_kps/2)), int(last_y+(num_kps/2)))
            cpatch_x = np.arange(int(last_x-(num_kps/2)), int(last_x+(num_kps/2)))
            # pts1 += list(zip(cpatch_x,cpatch_y))
            xs,ys = np.meshgrid(cpatch_x,cpatch_y)
            pts1 += list(zip(list(xs.flatten()), list(ys.flatten())))
                        
            curr_x = query_crater[0]-curr_xmin
            curr_y = query_crater[1]-curr_ymin
            cpatch_y = np.arange(int(curr_y-(num_kps/2)), int(curr_y+(num_kps/2)))
            cpatch_x = np.arange(int(curr_x-(num_kps/2)), int(curr_x+(num_kps/2)))
            # pts2 += list(zip(cpatch_x,cpatch_y))   
            xs,ys = np.meshgrid(cpatch_x,cpatch_y)
            pts2 += list(zip(list(xs.flatten()), list(ys.flatten())))      
            
            # Draw match lines between subplots
            if drawMatches:
                con = ConnectionPatch(xyA=(curr_x, curr_y), xyB=(last_x, last_y), coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="lime")
                # for curr,last in zip(pts2,pts1):
                #     con = ConnectionPatch(xyA=curr, xyB=last, coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="lime")
                ax2.add_artist(con)
                
    if not pts1:
        return None, None
        
    # print(cv2.findEssentialMat(pts1,pts2, focal=1.0, pp=(0,0), method=cv2.RANSAC, prob=0.999, threshold=1.0))
    import time
    # print(len(pts1), len(pts2))
    s = time.time()
    E,_ = cv2.findEssentialMat(np.array(pts1), np.array(pts2))
    # print(f'findEssentialMat: {time.time()-s}s')
    s = time.time()
    inliers, R, t, mask = cv2.recoverPose(E, np.array(pts1), np.array(pts2))
    # print(f'recoverPose: {time.time()-s}s')

    if drawMatches:
        plt.show()
        plt.close()
        
    return R,t

def simNav(img, lro_lonlats, craters_px, desc_model):
    # craters_px --> (center pixel X, center pixel Y, diameter in pixels)
    fsize = 3000//2 # x000/2 ==> x00 KM x x00 KM frame
    num_vis_craters = [] # history of number of crater detections per frame
    all_vis_crater_idxs = set()
    margin = 50 # pixel margin around crater diameter
    
    # Matching index
    match_threshold = 0.99 # Matches are 'cosine similarity >= this thresh'
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(512)
    index = faiss.IndexIDMap(index)
    # index = faiss.index_cpu_to_gpu(res, 0, index)
    index.reset()
    
    last_pos = None
    poses = {}
    correct_matches = 0
    incorrect_matches = 0
    index_ids = []
    frame_num = 0
    for lon,lat in lro_lonlats:
        print(f'---------- Frame {frame_num} ----------')
        
        x,y = latLongToPx(lat, lon)
        
        # Crop mosaic at frame
        frame = img[y-fsize:y+fsize, x-fsize:x+fsize]
        
        # Corner pixels and get visible craters
        xmin = x-fsize
        xmax = x+fsize
        ymin = y-fsize
        ymax = y+fsize
        vis_craters = [c for c in craters_px if checkCraterFits(xmin,xmax,ymin,ymax,c, margin)]
        vis_crater_idxs = [craters_px.index(c) for c in vis_craters] # ground truth crater index number
        print(f'Craters visible: {len(vis_craters)} - {vis_crater_idxs}')
        num_vis_craters.append(len(vis_craters))
        all_vis_crater_idxs.update(vis_crater_idxs)
        
        # Loop through visible craters in frame
        crater_detections = torch.ones((len(vis_craters), 3, 128, 128)) # Batch of preprocessed crater patches in current frame
        for i in range(len(vis_craters)):
            crater = vis_craters[i]
            x = crater[0]
            y = crater[1]
            diam_px = crater[2]        
            # Center the x/y point w.r.t the frame
            x = x-xmin
            y = y-ymin
            
            # Grab crater patch
            c_patch = frame[int(y-(diam_px/2)-margin):int(y+(diam_px/2)+margin), int(x-(diam_px/2)-margin):int(x+(diam_px/2)+margin)]
            
            # Resize and norm crater patch, store in detections batch
            c_patch = preprocess_image(c_patch)
            crater_detections[i,:] = c_patch
          
        # Visualize individual crater detections  
        # for _ in range(crater_detections.shape[0]):
        #     show_image_gray(crater_detections[_,:])
            
        # Compute embeddings (descriptors) for batch of visual craters
        embeddings = desc_model(crater_detections)
        
        # Batch-match set of landmark feature vectors over faiss database (index)
        feats = torch.nn.functional.normalize(embeddings)
        dists, ids = index.search(feats, 1)
        # for _ in range(dists.shape[0]):
        #     print(f'{vis_crater_idxs[_]}, {ids[_,0]}, {dists[_,0]}')
        
        # If feature vector not matched (ids == -1 OR dists <= match_threshold), add it to the database
        match_checks = torch.where(dists < match_threshold, -1, ids)
        non_matched_feats = feats[(match_checks == -1).nonzero()[:,0]]   
        # Get ground truth IDs of craters to act as index IDs 
        non_matched_ids = torch.tensor(vis_crater_idxs, dtype=torch.int64)[(match_checks == -1).nonzero()[:,0]]
        index.add_with_ids(non_matched_feats, non_matched_ids)

        if non_matched_ids.shape[0] >= 1:
            index_ids += list(non_matched_ids.numpy())
             
        # Get matches
        if frame_num != 0:
            query_ids = np.array(vis_crater_idxs)[(match_checks.numpy().squeeze(axis=-1) != -1).nonzero()]
            matched_ids = ids.numpy().squeeze(axis=-1)[(match_checks.numpy().squeeze(axis=-1) != -1).nonzero()]
            correspondance = list(zip(query_ids, matched_ids))
            for qid,mid in correspondance:
                if qid == mid:
                    correct_matches += 1
                else:
                    incorrect_matches += 1
                    
            # Estimate pose
            R,t = calcPose(img, fsize, craters_px, margin, last_pos, (lon,lat), correspondance, drawMatches=False)
            if R is not None:
                poses[frame_num] = {'R': R, 't': t}
            else:
                poses[frame_num] = {'R': poses[frame_num-1]['R'], 't': poses[frame_num-1]['t']}
            
        missed_matches = len(index_ids)-len(np.unique(index_ids))
        matching_acc = (correct_matches/(correct_matches+incorrect_matches+1e-6))*100
        recog_acc = (correct_matches/(correct_matches+incorrect_matches+missed_matches+1e-6))*100
            
        print(f'Craters seen:           {len(all_vis_crater_idxs)}')
        print(f'Index size:             {index.ntotal}')
        print(f'Missed matches:         {missed_matches}')
        print(f'Correct matches:        {correct_matches}')
        print(f'Incorrect matches:      {incorrect_matches}')
        print('Matching accuracy:      {:.2f}%'.format(matching_acc))
        print('Recognition accuracy:   {:.2f}%'.format(recog_acc))
        
        last_pos = (lon,lat)
        frame_num += 1
    
    return poses

def randomCraterInvariance(fullImg, craters_px, models):
    num_craters = 1000
    margin = 50
    crater_idxs = []
    distances = {}
    for model_type in list(models.keys()):
        distances[model_type] = {'bright': [], 'dark': [], 'rot1': [], 'rot2': [], 'rot3': []}
        model = FeatureExtractor.load_from_checkpoint(models[model_type]['path'], loss_func=model_type.lower(), miner=models[model_type]['miner']).cuda().eval()
        models[model_type]['model'] = model
    
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(512)
    index = faiss.index_cpu_to_gpu(res, 0, index)
    
    s = time.time()
    while len(crater_idxs) != num_craters:
        idx = np.random.choice(len(craters_px), 1, replace=False)
        crater = craters_px[idx[0]]
        x = crater[0]
        y = crater[1]
        diam_px = crater[2]  
        boundary = fullImg.shape[0]*0.25
        if y < boundary or y > fullImg.shape[0]-boundary:
            continue    
        c_patch = fullImg[int(y-(diam_px/2)-margin):int(y+(diam_px/2)+margin), int(x-(diam_px/2)-margin):int(x+(diam_px/2)+margin)]
        if (c_patch.shape[0] != c_patch.shape[1]) or (c_patch.shape[0] == 0 or c_patch.shape[1] == 0):
            continue 
        crater_idxs.append(idx)
        
        stock = augment_image(np.copy(c_patch), type='none').cuda()
        bup = augment_image(np.copy(c_patch), type='bup').cuda()
        bdown = augment_image(np.copy(c_patch), type='bdown').cuda()      
        rot1 = augment_image(np.copy(c_patch), type='rot').cuda()
        rot2 = augment_image(np.copy(c_patch), type='rot').cuda()
        rot3 = augment_image(np.copy(c_patch), type='rot').cuda()
        
        show_image_gray(stock)
        show_image_gray(bup)
        show_image_gray(bdown)        
        show_image_gray(rot1)
        show_image_gray(rot2)
        show_image_gray(rot3)
        
        # Stack up augmented images
        batch = torch.zeros((5,3,128,128)).cuda()
        batch[0,:] = bup
        batch[1,:] = bdown
        batch[2,:] = rot1
        batch[3,:] = rot2
        batch[4,:] = rot3
        
        for model_type in list(models.keys()):
            model = models[model_type]['model']
            index.reset()
            
            # 1.) extract 'stock' embedding, add to index
            stock_emb = model(stock[None,:])
            feats = torch.nn.functional.normalize(stock_emb)
            index.add(feats)
            
            # 2.) Extract augmented embeddings, match to index, collect dists
            batch_embs = model(batch)
            feats = torch.nn.functional.normalize(batch_embs)
            dists, ids = index.search(feats, 1)
            
            distances[model_type]['bright'].append(dists[0,0].cpu())
            distances[model_type]['dark'].append(dists[1,0].cpu())
            distances[model_type]['rot1'].append(dists[2,0].cpu())
            distances[model_type]['rot2'].append(dists[3,0].cpu())
            distances[model_type]['rot3'].append(dists[4,0].cpu())
            
        print(f'{time.time()-s}s')
                 
    # Plot results
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    # model_cols = ['firebrick', 'lightcoral', 'r', 'dodgerblue', 'skyblue', 'c']
    # # typestyles = ['-', '--', ':',]
    
    # # Bar chart
    # labels = ['+30% Brightness', '-30% Brightness', 'Random Rotations (3)']
    # means = {}
    # stds = {}
    # for model_type in list(models.keys()):
    #     means[model_type] = []
    #     stds[model_type] = []
        
    #     means[model_type].append(np.mean(distances[model_type]['bright']))
    #     stds[model_type].append(np.std(distances[model_type]['bright']))
    #     means[model_type].append(np.mean(distances[model_type]['dark']))
    #     stds[model_type].append(np.std(distances[model_type]['dark']))
    #     rots = np.array([distances[model_type]['rot1'], distances[model_type]['rot2'], distances[model_type]['rot3']])
    #     means[model_type].append(np.mean(rots))
    #     stds[model_type].append(np.std(rots))
    
    # width = 0.10
    # x = np.arange(len(labels))
    # fig, ax = plt.subplots()
    # bws = np.linspace(-0.25, 0.25, 6)
    # i=0
    # for model_type in list(models.keys()):
    #     rect = ax.bar(x+bws[i], means[model_type], width, label=model_type, color=model_cols[i], yerr=stds[model_type])
    #     ax.bar_label(rect, fmt='%.3f', padding=3)
    #     i+=1
    
    # ax.set_ylabel('Distance to Landmark (Cosine)')
    # ax.set_xticks(x, labels)
    
    # ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=6)
    
    # plt.show()

def main():
    
    '''
    LRO orbital period is 2 hours
    START_TIME                   = 2022-09-01T00:00:00.044
    STOP_TIME                    = 2022-09-11T00:00:07.963
    '''
    start = '2022-SEP-6 00:00:00'
    end   = '2022-SEP-6 02:00:00'
    freq = '10S'
    lro_lonlats = grabSpiceLonLats(start,end,freq)
    crater_lonlatdiams = grabCraterLonLats('/mnt/c/Users/TJ/Downloads/moon_craters/LolaLargeLunarCraterCatalog.csv')
    craters_px = convertCraterDatabasePixels(crater_lonlatdiams)
    craters_px.sort()
    
    print(f"LRO Trajectory Frames: {len(lro_lonlats)}")
    print(f"Crater Database Size: {len(crater_lonlatdiams)}")
    
    # Band1 for visualization (tif is grayscale)
    s = time.time()
    fullImg = np.array(ds.GetRasterBand(1).ReadAsArray())
    print(f'Full raster loaded ({time.time()-s}s)...')

    # visualizeTrajectory(fullImg, lro_lonlats[:400])
    # quit()
    # visualizeCraters(fullImg, crater_lonlatdiams)
    
    # Grab trained descriptor model paths, select one, and load model
    # model_type = 'NTXent'
    # print(f'Descriptor Model Selection: {model_type}')
    # model = FeatureExtractor.load_from_checkpoint(model_paths[model_type]['path'], loss_func=model_type.lower(), miner=model_paths[model_type]['miner']).eval()
    
    # Return poses in KITTI format
    # poses = simNav(fullImg, lro_lonlats, craters_px, model)
    # write_posefile(poses, 'ntxent_300km_m50_thresh0.99_poses.txt')
    
    randomCraterInvariance(fullImg, craters_px, model_paths)


if __name__ == '__main__':
    main()