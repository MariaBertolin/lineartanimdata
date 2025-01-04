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

def preprocess(img): # no s'usa
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3) # imatge amb soroll reduit, conservem bordes
    highpass = img.astype(int) - blurred.astype(int) # resaltem els bordes i detalls, eliminem elements de baixa freqüència
    highpass = highpass.astype(float) / 128.0 # escalem els valors
    highpass /= np.max(highpass) # normalitzem

    new = np.zeros((512, 512, 4), dtype=float) # generem nova imatge de 512x512 RGBA
    new[0:h,0:w,0:c] = highpass
    return new
    
def detect_and_refine_lines(img): # no s'usa
    if img.shape[-1] == 4:  # Imatge amb canal alfa
        alpha_channel = img[:, :, 3] # create a mask of alpha == 0
        # img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) -> S'OBTENEN MILLORS RESULTATS AMB LA IMATGE ORIGINAL
        img[alpha_channel == 0] = 255  # Zones transparents -> blanc
    else
        raise ValueError('La imatge proporcionada no té canal alfa, no cumpleix amb el format requerit')
    # Detecció de bordes
    img = cv2.medianBlur(img, 3) # Tot i que Canny preprocessa la imatge per defecte (aplica Gaussiana), he vist que posant un filtre de mediana obté millors resultats
    edges = cv2.Canny(img, thres_low=50, thres_upper=100)  # A menor interval, major detall 

    # Suavitzar les línies, tot i que el millor resultat és directament l'output de canny
    kernel = np.ones((2, 2), np.uint8)  # Kernel más pequeño para menos grosor
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Mantener suavidad sin engrosar demasiado
    smoothed = cv2.medianBlur(dilated, 5)  # Eliminar ruido
    return edges

def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0
    # Verifica si pred tiene más dimensiones de las esperadas
    if pred.ndim > 2:
        pred = np.amax(pred, axis=0)  # Redueix a 2D amb el máxim en el primer eix

    pred[pred < thresh] = 0
    pred = 1 - pred # les línies les volem negres i la resta blanc!
    pred *= 255 # 1=255 (blanc)
    pred = np.clip(pred, 0, 255).astype(np.uint8) # clipping
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":
    args = parse_args()

    SCENE_PATH = args.input

    inputpath = SCENE_PATH + "/out_clustercolor" 
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
            
            # PRE PROCESS with Median Blur Filter
            img_mbf = cv2.medianBlur(img, 5)
            
            # DETECT EDGES
            t_l = 50; t_u = 100 # 100,200

            # with median blur filter
            lines = cv2.Canny(img_mbf, t_l, t_u) # millors resultats
            
            # postprocess
            output = postprocess(lines, thresh=0.1, smooth=False)
         
            cv2.imwrite(os.path.join(outputpath, filename), output)
