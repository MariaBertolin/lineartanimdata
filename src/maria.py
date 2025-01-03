import argparse
from google.colab import output
import numpy as np
import torch
import cv2
# from model import SketchKeras
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="", help="input image file")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="output image file")
    # parser.add_argument("--weight", "-w", type=str, default="weights/model.pth", help="weight file")
    return parser.parse_args()


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(float) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 4), dtype=float)
    ret[0:h,0:w,0:c] = highpass
    return img
    
def detect_and_refine_lines(img):
    # Manejar transparencia si la imagen tiene un canal alfa
    if img.shape[-1] == 4:  # Imagen con canal alfa
      # create a mask of alpha == 0
        alpha_channel = img[:, :, 3]
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        img[alpha_channel == 0] = 255  # Zonas transparentes -> blanco
    
    # Aplicar detección de bordes
    img = cv2.medianBlur(img, 3)
    edges = cv2.Canny(img, threshold1=50, threshold2=100)  # Reducir los umbrales para más detalle

    # Suavizar las líneas
    kernel = np.ones((2, 2), np.uint8)  # Kernel más pequeño para menos grosor
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Mantener suavidad sin engrosar demasiado
    smoothed = cv2.medianBlur(dilated, 5)  # Eliminar ruido

    return edges

def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0
    # Verifica si pred tiene más dimensiones de las esperadas
    if pred.ndim > 2:
        pred = np.amax(pred, axis=0)  # Reduce a 2D tomando el máximo en el primer eje

    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":
    args = parse_args()

    SCENE_PATH = args.input

    inputpath = SCENE_PATH + "/out_clustercolor" 
    #inputpath = SCENE_PATH+"/out_foreground"
    #outputpath = SCENE_PATH+"/out_sketch_fromforeground/"
    outputpath = SCENE_PATH + "/out_sketch_fromclustercolor/" 
    
    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process(" + inputpath + "/" + filename + ", " + outputpath + ")")
            
            img = cv2.imread(os.path.join(inputpath, filename), cv2.IMREAD_UNCHANGED)
            assert img is not None, "file could not be read, check with os.path.exists()"

            if img.shape[-1] == 4:  # Imagen con canal alfa
                # create a mask of alpha == 0
                alpha_channel = img[:, :, 3]
                img[alpha_channel == 0] = 255  # Zonas transparentes -> blanco
            
            # PRE PROCESS with median blur filter
            img_mbf = cv2.medianBlur(img, 5)
            
            # DETECT EDGES
            a1 = 50; a2 = 100 # 100,200

            # with median blur filter
            lines = cv2.Canny(img_mbf, a1, a2) # millors resultats
            
            # postprocess
            output = postprocess(lines, thresh=0.1, smooth=False)
         
            cv2.imwrite(os.path.join(outputpath, filename), output)
