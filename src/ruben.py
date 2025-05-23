import argparse
import numpy as np
import torch
import cv2
from model import SketchKeras
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="", help="input image file")
    parser.add_argument(
        "--output", "-o", type=str, default="output.jpg", help="output image file"
    )
    parser.add_argument(
        "--weight", "-w", type=str, default="weights/model.pth", help="weight file"
    )
    return parser.parse_args()


def preprocess(img):
    h, w, c = img.shape
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    highpass = img.astype(int) - blurred.astype(int)
    highpass = highpass.astype(np.float) / 128.0
    highpass /= np.max(highpass)

    ret = np.zeros((512, 512, 3), dtype=np.float)
    ret[0:h,0:w,0:c] = highpass
    return ret


def postprocess(pred, thresh=0.18, smooth=False):
    assert thresh <= 1.0 and thresh >= 0.0

    pred = np.amax(pred, 0)
    pred[pred < thresh] = 0
    pred = 1 - pred
    pred *= 255
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    if smooth:
        pred = cv2.medianBlur(pred, 3)
    return pred


if __name__ == "__main__":
    args = parse_args()

    model = SketchKeras().to(device)

    if len(args.weight) > 0:
        model.load_state_dict(torch.load(args.weight))
        print(f"{args.weight} loaded..")

    SCENE_PATH = args.input

    inputpath = SCENE_PATH+"/out_clustercolor"
    #inputpath = SCENE_PATH+"/out_foreground"
    #outputpath = SCENE_PATH+"/out_sketch_fromforeground/"
    outputpath = SCENE_PATH+"/out_sketch_fromclustercolor/"
    
    

    if not os.path.exists(outputpath):
       os.makedirs(outputpath)

    for filename in sorted(os.listdir(inputpath)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):# and filename=="00066.png":
            print("process("+inputpath+"/"+filename+", "+outputpath+")")
            
            img_orig = cv2.imread(os.path.join(inputpath, filename))
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
            x = img.reshape(1, *img.shape).transpose(3, 0, 1, 2)
            x = torch.tensor(x).float()
            
            # feed into the network
            with torch.no_grad():
                pred = model(x.to(device))
            pred = pred.squeeze()
            
            # postprocess
            output = pred.cpu().detach().numpy()
            output = postprocess(output, thresh=0.1, smooth=False) 
            output = output[:new_height, :new_width]

            #resize to original
            output = cv2.resize(output, (img_orig.shape[1], img_orig.shape[0]))
            
            
            cv2.imwrite(os.path.join(outputpath, filename), output)
