import matplotlib.pyplot as plt
#import shapely.geometry
import cv2
import numpy as np
#import util_contours
import os
import traceback
#import facial_landmarks
import sys

def removeBlack(img):
    color = (0,0,0)
    mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result

def clustercolor(image):
    clusters=10
    rounds=1
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    #samples = np.zeros([h*w,4], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            #if not (image[x][y][0] == 0 and image[x][y][1] == 0 and image[x][y][2] == 0):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,clusters, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), rounds, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

    return res

def clustercolor_scene(SCENE_PATH):
    inputpathForeground = SCENE_PATH+"/out_foreground"
    outputpath = SCENE_PATH+"/out_clustercolor/"

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpathForeground)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpathForeground+"/"+filename+", "+outputpath+")")
            
            foreground_img = cv2.imread(os.path.join(inputpathForeground, filename), cv2.IMREAD_UNCHANGED)
            #foreground_img = cv2.imread(os.path.join(inputpathForeground, filename))
            
            b_channel, g_channel, r_channel, a_channel = cv2.split(foreground_img)
            foreground_img_noalpha = cv2.merge((b_channel, g_channel, r_channel))
            
            assert foreground_img is not None, "file could not be read, check with os.path.exists()"
           
            clustered_img = clustercolor(foreground_img_noalpha)

            b_channel, g_channel, r_channel = cv2.split(clustered_img)
            clustered_img = cv2.merge((b_channel, g_channel, r_channel, a_channel))

            cv2.imwrite(os.path.join(outputpath, filename), clustered_img)
            
if __name__ == "__main__":
    SCENE_PATH = sys.argv[1]

    clustercolor_scene(SCENE_PATH)                