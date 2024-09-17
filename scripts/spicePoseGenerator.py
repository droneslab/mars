import spiceypy as sp
import numpy as np
import csv
import pandas as pd
import glob
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

def visualizeTraj(poses):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(poses[:,0], poses[:,1], poses[:,2])
    plt.show()

print('----- Loading Kernels -----')
kernel_list = glob.glob('../data/spice_kernels/**/*')
for filename in kernel_list:
    if 'Identifier' in filename or 'lbl' in filename:
        continue
    # Load kernel
    print("\tLoading kernel: {}".format(filename))
    sp.furnsh(filename)

'''
LRO orbital period is 2 hours
START_TIME                   = 2022-09-01T00:00:00.044
STOP_TIME                    = 2022-09-11T00:00:07.963
'''
start = '2022-SEP-6 00:00:00'
end   = '2022-SEP-6 06:00:00'

# '''
# Frequency (case sensitive)
# H     - hours (e.g. 5H)
# T,min - minutes (e.g. 5T)
# S     - seconds
# L     - ms
# '''
freq = '10S'

epochs = list(pd.date_range(start=start, end=end, freq=freq))
epochs = [x.to_pydatetime() for x in epochs]
s = epochs[0]
e = epochs[-1]
epochs = [x.strftime("%Y-%b-%d %H:%M:%S.%f") for x in epochs]

delt = str(e-s)
dH = str(e-s).split(':')[0]
pose_out_file = f'../data/pose_csv/kitti_LROCWAC_{s.strftime("%y%b%d_%H%M%S")}_{dH}H_{freq}_poses.csv'
latlon_out_file = f'../data/lonlat_csv/LROCWAC_{s.strftime("%y%b%d_%H%M%S")}_{dH}H_{freq}_lonlat.csv'

print('----- Timeframe -----')
print('\tStart: {}'.format(start))
print('\tEnd:   {}'.format(end))
print('\tFreq:  {}'.format(freq))

i = 0
# poses = []
lat_lons = []
kitti_poses = []
print('----- Calculating Poses [x y z q0 q1 q2 q3] -----')
for epoch in epochs:
    i += 1

    et = sp.str2et(epoch)
    print('\t{}'.format(sp.et2utc(et, "C", 2)))
    
    # [x y z] LRO Position (SPK). MOON_PA --> Lunar Principal Axis
    lro_pos,_ = sp.spkpos('LRO', et, 'MOON_PA', 'LT+S', 'MOON')
    
    # Store lat/lon translation
    t = SkyCoord(x=lro_pos[0], y=lro_pos[1], z=lro_pos[2], representation_type='cartesian')
    lat_lons.append((t.spherical.lon.degree, t.spherical.lat.degree))

    # 3x3 LRO_LROCWAC Rotation
    wac_ori = sp.pxform("LRO_LROCWAC", "IAU_MOON", et)
    # 3x3 LRO_LROCWAC Rotation --> Quaternion
    # wac_q = sp.m2q(wac_ori)
    
    # pose = np.array([lro_pos[0], lro_pos[1], lro_pos[2], wac_q[0], wac_q[1], wac_q[2], wac_q[3]])
    # poses.append(pose)
    
    '''
    Each row of the file contains the first 3 rows of a 4x4 homogeneous pose matrix flattened into one line.
    It means that this matrix:
        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz
        0   0   0   1
    is represented in the file as a single row: 
        r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
    '''
    pose = np.array([wac_ori[0][0], wac_ori[0][1], wac_ori[0][2], lro_pos[0], wac_ori[1][0], wac_ori[1][1], wac_ori[1][2], lro_pos[1], wac_ori[2][0], wac_ori[2][1], wac_ori[2][2], lro_pos[2]])
    kitti_poses.append(pose)
    
# visualizeTraj(np.array(poses))

############################ UNUSED FOR NOW
    # 3x3 Sun Rotation
#     sun_ori = sp.pxform("ORX_SUN_PLANE_OF_SKY", "IAU_BENNU", et)

    # 3x3 Sun Rotation --> Quaternion
#     sun_q = sp.m2q(sun_ori)
############################
      
print('----- Saving Poses to CSV -----')
with open(pose_out_file, 'w+') as f:
    writer = csv.writer(f, delimiter=',')
    for pose in kitti_poses:
        writer.writerow(pose)
print("\tWrote {} poses to: ".format(len(kitti_poses)) + pose_out_file.split('/')[-1])

# print('----- Saving LON/LATs to CSV -----')
# with open(latlon_out_file, 'w+') as f:
#     writer = csv.writer(f, delimiter=',')
#     for latlon in lat_lons:
#         writer.writerow(latlon)
# print("\tWrote {} lonlats to: ".format(len(lat_lons)) + latlon_out_file.split('/')[-1])
