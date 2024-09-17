import sys
sys.path.append('../src/')
from model import ResNext
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import csv
from osgeo import gdal, osr
from geoRender import show_image_gray, grabSpiceLonLats, grabCraterLonLats, convertCraterDatabasePixels, parseModelCkpts
from eval import augment_image
import umap
import umap.plot

ds = gdal.Open('/mnt/c/Users/TJ/Downloads/moon_craters/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')
xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +R=1737400 +no_defs')
start = '2022-SEP-6 00:00:00'
end   = '2022-SEP-6 02:00:00'
freq = '10S'
lro_lonlats = grabSpiceLonLats(start,end,freq)
crater_lonlatdiams = grabCraterLonLats('/mnt/c/Users/TJ/Downloads/moon_craters/LolaLargeLunarCraterCatalog.csv')
craters_px = convertCraterDatabasePixels(crater_lonlatdiams)
craters_px.sort()
print('Loading raster...')
s = time.time()
fullImg = np.array(ds.GetRasterBand(1).ReadAsArray())
print(f'Raster loaded ({time.time()-s}s)...')

models = parseModelCkpts()
model_type = 'NTXent'
model = ResNext.load_from_checkpoint(models[model_type]['path'], loss_func=model_type.lower(), miner=models[model_type]['miner']).cuda().eval()

while True:
    embeddings = []
    ys = []
    for i in range(10):
        idx = np.random.choice(len(craters_px), 1, replace=False)
        crater = craters_px[idx[0]]
        x = crater[0]
        y = crater[1]
        diam_px = crater[2]  
        boundary = fullImg.shape[0]*0.25
        if y < boundary or y > fullImg.shape[0]-boundary:
            continue
        margin = 50
        c_patch = fullImg[int(y-(diam_px/2)-margin):int(y+(diam_px/2)+margin), int(x-(diam_px/2)-margin):int(x+(diam_px/2)+margin)]
        if (c_patch.shape[0] != c_patch.shape[1]) or (c_patch.shape[0] == 0 or c_patch.shape[1] == 0):
            continue

        box_jitter = np.random.randint(low=15, high=50, size=2)
        x = x - box_jitter[0]
        y = y - box_jitter[1]
        c_patch_trans = fullImg[int(y-(diam_px/2)-margin):int(y+(diam_px/2)+margin), int(x-(diam_px/2)-margin):int(x+(diam_px/2)+margin)]

        stock = augment_image(c_patch, type='none').cuda()
        trans = augment_image(c_patch_trans, type='none').cuda()
        bup25 = augment_image(c_patch, type='bup25').cuda()
        bdown25 = augment_image(c_patch, type='bdown25').cuda()
        bup50 = augment_image(c_patch, type='bup50').cuda()
        bdown50 = augment_image(c_patch, type='bdown50').cuda()     
        bup75 = augment_image(c_patch, type='bup75').cuda()
        bdown75 = augment_image(c_patch, type='bdown75').cuda()           
        rot90 = augment_image(c_patch, type='rot90').cuda()
        rot180 = augment_image(c_patch, type='rot180').cuda()
        rot270 = augment_image(c_patch, type='rot270').cuda()
        rot1 = augment_image(c_patch, type='ranRot').cuda()
        rot2 = augment_image(c_patch, type='ranRot').cuda()
        rot3 = augment_image(c_patch, type='ranRot').cuda()
        pers = augment_image(c_patch, type='pers').cuda()
        gauss = augment_image(c_patch, type='gauss').cuda()
        shot = augment_image(c_patch, type='shot').cuda()
        
        batch = torch.zeros((15,3,128,128)).cuda()
        batch[0,:] = stock
        batch[1,:] = bup25
        batch[2,:] = bdown25
        batch[3,:] = bup50
        batch[4,:] = bdown50
        batch[5,:] = bup75
        batch[6,:] = bdown75
        batch[7,:] = rot90
        batch[8,:] = rot180
        batch[9,:] = rot270
        batch[10,:] = rot1
        batch[11,:] = rot2
        batch[12,:] = rot3
        batch[13,:] = trans
        batch[14,:] = pers
        # batch[15,:] = gauss
        # batch[16,:] = shot
        
        feats = model(batch)
        embeddings.append(feats.detach().cpu().numpy())
        
        # ys += [i]*17
        # ys += list(range(17))
        # ys += [0,1,2,1,2,1,2,3,3,3,4,4,4,5,6]

    embeddings = np.concatenate(embeddings, axis=0)
    ys = np.array(ys)
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine').fit(embeddings)
    umap.plot.points(mapper, labels=ys, theme='fire', show_legend=True)
    plt.show()