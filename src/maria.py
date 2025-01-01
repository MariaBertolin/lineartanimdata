import argparse
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
    return ret
    
def detect_and_refine_lines(img):
    # Manejar transparencia si la imagen tiene un canal alfa
    if img.shape[-1] == 4:  # Imagen con canal alfa
        alpha_channel = img[:, :, 3]
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        img[alpha_channel == 0] = 255  # Zonas transparentes -> blanco
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes
    edges = cv2.Canny(img, threshold1=150, threshold2=250)  # Reducir los umbrales para más detalle

    # Suavizar las líneas
    kernel = np.ones((2, 2), np.uint8)  # Kernel más pequeño para menos grosor
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Mantener suavidad sin engrosar demasiado
    smoothed = cv2.medianBlur(dilated, 5)  # Eliminar ruido

    return dilated

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
            
            img_orig = cv2.imread(os.path.join(inputpath, filename), cv2.IMREAD_UNCHANGED)
            assert img_orig is not None, "file could not be read, check with os.path.exists()"

            # resize
            height, width = float(img_orig.shape[0]), float(img_orig.shape[1])
            if width > height:
                new_width, new_height = (512, int(512 / width * height))
            else:
                new_width, new_height = (int(512 / height * width), 512)
            img = cv2.resize(img_orig, (new_width, new_height))
            
            # preprocess
            img = preprocess(img)
            img_np = img[:, :, :3]  # Extraer como NumPy array
            img_np = (img_np * 255).astype(np.uint8)  # Escalar a rango válido
            
            # Detect and refine lines
            lines = detect_and_refine_lines(img_np)
            
            # postprocess
            output = postprocess(lines, thresh=0.1, smooth=False) 
            
            # Verifica que `output` sea 2D antes de indexarlo
            if output.ndim == 2:
                output = output[:new_height, :new_width]
            else:
                raise ValueError(f"Unexpected output dimensions: {output.shape}")

            #resize to original
            output = cv2.resize(output, (img_orig.shape[1], img_orig.shape[0]))
            
            cv2.imwrite(os.path.join(outputpath, filename), output)
