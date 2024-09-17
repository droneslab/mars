import cv2
import glob
import numpy as np
from alive_progress import alive_bar
import sys
from pathlib import Path

stock_image_path = sys.argv[1]
ext = sys.argv[2]
dset_savepath = sys.argv[3]
Path(dset_savepath).mkdir(parents=True, exist_ok=True)

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

'''
HIRISE Style:
    1. 90 degrees clockwise rotation
    2. 180 degrees clockwise rotation
    3. 270 degrees clockwise rotation
    4. Horizontal flip
    5. Vertical flip
    6. Random brightness adjustment
'''
if __name__ == '__main__':
    image_paths = sorted(glob.glob(stock_image_path + '/*.' + ext))
    
    with alive_bar(len(image_paths)) as bar:
        for i in range(len(image_paths)):
            path = image_paths[i]
            img = cv2.imread(path)
            r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            r180 = cv2.rotate(img, cv2.ROTATE_180)
            r270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            hf = cv2.flip(img, 1)
            vf = cv2.flip(img, 0)
            bValue = np.random.randint(-30, 31)
            ill = change_brightness(img, bValue)
            
            cv2.imwrite(dset_savepath + f'{i}.png', img)
            cv2.imwrite(dset_savepath + f'{i}_r90.png', r90)
            cv2.imwrite(dset_savepath + f'{i}_r180.png', r180)
            cv2.imwrite(dset_savepath + f'{i}_r270.png', r270)
            cv2.imwrite(dset_savepath + f'{i}_hf.png', hf)
            cv2.imwrite(dset_savepath + f'{i}_vf.png', vf)
            cv2.imwrite(dset_savepath + f'{i}_ill.png', ill)
            
            bar()