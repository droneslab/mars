import time
from osgeo import gdal, osr
import numpy as np
import sys
sys.path.append('../')
from scripts.geoRender import show_image_gray, grabSpiceLonLats, grabCraterLonLats, convertCraterDatabasePixels, preprocess_image
import cv2
from alive_progress import alive_bar
from eval import augment_image

def tToN(tensor):
    tensor = (tensor.permute(1, 2, 0)*255).numpy()
    return tensor.astype(np.uint8)

dataset_loc = '/home/tj/data/lunarCraters/images_stock/'

ds = gdal.Open('/home/tj/data/lunar_models/mosaics/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')
xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +R=1737400 +no_defs')
start = '2022-SEP-6 00:00:00'
end   = '2022-SEP-6 02:00:00'
freq = '10S'
lro_lonlats = grabSpiceLonLats(start,end,freq)

large_crater_lonlatdiams = grabCraterLonLats('/home/tj/data/lunar_models/crater_dbs/LolaLargeLunarCraterCatalog.csv')
large_craters_px = convertCraterDatabasePixels(large_crater_lonlatdiams)
large_craters_px.sort()
print(len(large_craters_px))

# small_crater_lonlatdiams = grabCraterLonLats('/mnt/c/Users/TJ/Downloads/moon_craters/lunar_crater_database_robbins_2018.csv', lonLine=2, latLine=1, diamLine=5)
# small_craters_px = convertCraterDatabasePixels(small_crater_lonlatdiams)
# small_craters_px.sort()
# print(len(small_craters_px))

print('Loading raster...')
s = time.time()
fullImg = np.array(ds.GetRasterBand(1).ReadAsArray())
print(f'Raster loaded ({time.time()-s}s)...')

margin = 50
i = 0
with alive_bar(len(large_craters_px)) as bar:
    for crater in large_craters_px:
        x = crater[0]
        y = crater[1]
        diam_px = crater[2]  
        boundary = fullImg.shape[0]*(1/4)
        if y < boundary or y > fullImg.shape[0]-boundary:
            bar()
            continue    
        c_patch = fullImg[int(y-(diam_px/2)-margin):int(y+(diam_px/2)+margin), int(x-(diam_px/2)-margin):int(x+(diam_px/2)+margin)]
        if (c_patch.shape[0] != c_patch.shape[1]) or (c_patch.shape[0] == 0 or c_patch.shape[1] == 0):
            bar()
            continue
        
        c_patch = cv2.resize(c_patch, (128,128))
        cv2.imwrite(dataset_loc+f'{i}.png', c_patch)
        
        i += 1
        bar()

