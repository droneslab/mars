import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
from geoRender import *
import argparse
from astropy.coordinates import SkyCoord
from osgeo import gdal, osr

if args.lowres:
    ds = gdal.Open('/mnt/c/Users/TJ/Downloads/WAC_GLOBAL_E000N0000_004P.tiff')
else:
    ds = gdal.Open('/mnt/c/Users/TJ/Downloads/moon_craters/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif')

''' NOTATION 
crs        ==> GeoTiff CRS (PROJCRS["SimpleCylindrical MOON"])
crsGeo     ==> Moon 2000 CRS
x,y        ==> pixel coordinates
posX,posY  ==> crs coordinates
lat,long   ==> crsGeo coordinates
crater database ==> extent -180 to 180


GeoTiff info given by: 'gdalinfo /mnt/c/Users/TJ/Downloads/moon_craters/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif'
'''
xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
# get CRS from dataset 
crs = osr.SpatialReference()
crs.ImportFromWkt(ds.GetProjectionRef())
# Proj4 format of 'Moon 2000': https://spatialreference.org/ref/iau2000/30100/ (ALSO FROM QGIS)
crsGeo = osr.SpatialReference()
crsGeo.ImportFromProj4('+proj=longlat +R=1737400 +no_defs')

# filename = sys.argv[1]

# traj_R = 0.0
# traj_t = np.array([[-0.34690893], [-0.84487322], [0.40723879]])

# i = 0
# xs = []
# ys = []
# with open(filename, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         pose = line.split(' ')
#         pose = [float(x) for x in pose]
#         R = np.array([[pose[0],pose[1],pose[2]],
#                       [pose[4],pose[5],pose[6]],
#                       [pose[8],pose[9],pose[10]]])
#         t = np.array ([[pose[3]],
#                        [pose[7]],
#                        [pose[11]]])
#         t = t.T
        
#         if i == 0:
#             traj_R = np.copy(R)
#             traj_t = traj_t + np.matmul(t, traj_R)
#             # traj_t = np.copy(t)
#         else:
#             traj_R = np.matmul(R,traj_R)
#             traj_t = traj_t + np.matmul(t, traj_R)
#             # traj_t = traj_t + t
                
#         x = traj_t[0,0]
#         y = -traj_t[0,1]
#         xs.append(x)
#         ys.append(y)
#         i += 1

# plt.figure()
# plt.scatter(xs,ys)
# plt.show()

def get_poses(csv_f):
    poses = {}
    with open(csv_f, 'r') as f:
        lines = f.readlines()
        frame_num = 1
        for line in lines:
            pose = line.split(' ')
            pose = [float(x) for x in pose]
            R = np.array([[pose[0],pose[1],pose[2]],
                        [pose[4],pose[5],pose[6]],
                        [pose[8],pose[9],pose[10]]])
            t = np.array([[pose[3]], [pose[7]], [pose[11]]])
            poses[frame_num] = {'R': R, 't': t}
            frame_num += 1
    return poses

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
    
    '''
    FLIP TO BEG OF DEM OCCURS AT LRO POSE 577
    lon,lat,x,y
    -175.86719727019465 83.98979904119824 16 24
    '''
    
    poses = get_poses(args.csv)
    num_poses = len(list(poses.keys()))
    
    gt = np.zeros((len(lro_lonlats), 3))
    i = 0
    lro_pxs = []
    for lon,lat in lro_lonlats:
        x,y,z = latLongToXYZ(lat,lon)
        gt[i,0] = x
        gt[i,1] = y
        gt[i,2] = z
        
        pxx, pxy = latLongToPx(lat, lon)
        lro_pxs.append((pxx,pxy))
        
        i += 1
        
    scales = []
    for i in range(1,len(lro_pxs)):
        tLast = np.array(lro_pxs[i-1])
        tCurr = np.array(lro_pxs[i])
        scale = np.linalg.norm(tCurr - tLast)
        # scales.append(tCurr-tLast)
        scales.append(scale)   
    
    # Calculate trajectory w.r.t starting LRO pose pixel location
    # lro_startx, lro_starty = latLongToPx(lro_pxs[0][1], lro_lonlats[0][0])
    lro_flipx, lro_flipy = (16,24)
    
    current_point = np.array([lro_pxs[0][0], lro_pxs[0][1], 0])
    nav_px = []
    nav_lonlats = []
    nav_cart = np.zeros((num_poses, 3))
    for frame_num in list(poses.keys()):
        
        # if frame_num >= 577:
        #     current_point = np.array([lro_flipx, lro_flipy, 0])
            
        nav_px.append(current_point)
        
        lat,lon = pxToLatLong(current_point[0], current_point[1])
        # if frame_num >= 570:
            # print(lat,lon, lro_lonlats[frame_num-1])
            # lat = -lat
            # lon = -lon
        # lat = -90 if lat < -90 else lat
        # lat = 90 if lat > 90 else lat
        nav_lonlats.append((lon,lat))
        
        x,y,z = latLongToXYZ(lat,lon)
        if frame_num >= 570:
            nav_cart[frame_num-1,0] = -x
            nav_cart[frame_num-1,1] = -y
        else:
            nav_cart[frame_num-1,0] = x
            nav_cart[frame_num-1,1] = y
        nav_cart[frame_num-1,2] = z
    
        scale = scales[frame_num-1]
    
        pose = poses[frame_num]
        t = pose['t']
        t = t.reshape((3,))
        if frame_num >= 577:
            t[0] = -t[0]*scale
            t[1] = -t[1]*scale
        else:
            t[0] = -t[0]*scale
            t[1] = -t[1]*scale
            
        if abs(t[2]*1000) <= 1:
            current_point = current_point + t
        else:
            current_point = current_point
                    
    nav_px = np.array(nav_px)
    
    # Pixel path plot
    print('-----')
    print('Pixel path [XZ]')
    plt.figure()
    plt.plot(nav_px[:,0], nav_px[:,2])
    plt.show()
    print('Pixel path [XY]')
    plt.figure()
    plt.plot(nav_px[:,0], nav_px[:,1])
    plt.show()
    
    # Cartesian spherical XYZ
    print('Cartesian spherical XYZ')
    visualizeXYZ(nav_cart, gt[:num_poses,:])
    print('Cartesian spherical XZ')
    plt.figure()
    plt.scatter(gt[:num_poses,0], gt[:num_poses,2])
    plt.scatter(nav_cart[:,0], nav_cart[:,2])
    plt.show()
    
    # Lat/lon on cylindrical
    print('Lat/long cylindrical')
    fullImg = np.array(ds.GetRasterBand(1).ReadAsArray())    
    visualizeTrajectory(fullImg, nav_lonlats, lro_lonlats)
        
    
    
    # traj_R = 0.0
    # traj_t = np.array([[start_px[0]], [start_px[1]], [0.0]])
    # cart_poses = np.zeros((num_poses, 3))
    # for frame_num in list(poses.keys()):
    #     pose = poses[frame_num]
    #     R = pose['R']
    #     t = pose['t']
        # t = t.T
        # if frame_num == 1:
        #     traj_R = np.copy(R)
        #     traj_t = traj_t + t
        # else:
        #     traj_R = np.matmul(R,traj_R)
        #     traj_t = traj_t + np.matmul(t, traj_R)
        # print(traj_t)
        # print(t)

        
    #     lat,lon = pxToLatLong(xy[0],xy[1])
    #     x,y,z = latLongToXYZ(lat,lon)
    #     cart_poses[frame_num-1,0] = x
    #     cart_poses[frame_num-1,1] = y
    #     cart_poses[frame_num-1,2] = z
        
    # visualizeXYZ([cart_poses])
    
    
    
    
    
    


if __name__ == '__main__':
    main()
