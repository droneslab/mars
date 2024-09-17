from osgeo import gdal, osr
import sys
sys.path.append('../')
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

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

def convertCraterDatabasePixels(lonlatdiams, raster_w):
    craters_px = []
    for line in lonlatdiams:
        lon = line[0]
        lat = line[1]
        diam_km = line[2]
        diam_m = diam_km * 1000
        diam_px = diam_m/100
        
        global_w = 109164
        scale_fac = global_w/raster_w
        diam_px = diam_px/scale_fac
        x,y = latLongToPx(lat,lon)    
        craters_px.append( (x,y,diam_px) )
    return craters_px

def convertCratersDiamLatLon(craters_px, raster_w):
    craters_diamLatLon = []
    for c in craters_px:
        x = c[0]
        y = c[1]
        diam_px = c[2] # Diameter px in current raster
        rad = diam_px/2
        
        l = x-rad
        if l < 0:
            l = raster_w-abs(l)
        r = x+rad
        if r > raster_w:
            r = r-raster_w
        l_latlon = pxToLatLong(l,y)
        r_latlon = pxToLatLong(r,y)
        c_latlon = pxToLatLong(x,y)
        craters_diamLatLon.append( (c_latlon[0], c_latlon[1], l_latlon[1], r_latlon[1]) )
    return craters_diamLatLon


ds = gdal.Open('/home/tj/data/lunar_models/mosaics/WAC_GLOBAL_E000N0000_064P.tiff')
xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +R=1737400 +no_defs')
print('Loading raster...')
s = time.time()
fullImg = np.array(ds.GetRasterBand(1).ReadAsArray())
print(f'Raster loaded ({time.time()-s}s)...')
crater_lonlatdiams = grabCraterLonLats('/home/tj/data/lunar_models/crater_dbs/LolaLargeLunarCraterCatalog.csv')
craters_px = convertCraterDatabasePixels(crater_lonlatdiams, fullImg.shape[1])
craters_px.sort()

import pandas as pd
craters_new_latlons = convertCratersDiamLatLon(craters_px, fullImg.shape[1])
df = pd.DataFrame(craters_new_latlons, columns=['c_lat', 'c_lon', 'l_lon', 'r_lon'])

# df = pd.DataFrame(craters_px, columns=['x','y','diam_px'])
csv_out = '/home/tj/data/lunar_models/crater_dbs/crater_latlons_64p.csv'
df.to_csv(csv_out, index=False)
    