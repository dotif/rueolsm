#import v4l2ctl 
import PIL
import torch 
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import torchvision.transforms as T
import os
import cv2
from pygrabber.dshow_graph import FilterGraph



def getCameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    # list of camera device
    
    return devices

#def getCameras():
#    vc = v4l2ctl.V4l2Device.iter_devices()
#    devices = {}
#    index = len(list(vc))-1
#    for dev in vc:        
#        camName = dev.name.split(':')[1].strip() 
#        devices[camName]=index 
#        index -= 1
#    return devices
 
def selectCamera():
    devices = getCameras()
    resValidas = []
    for idx, dev in enumerate(devices):
        print(idx,' : ',dev)
        resValidas.append(idx)

    while True:
        val = input("Selecciona la camara con la que vas a trabajar: ")
        try: 
            res = int(val)
            if res in resValidas:
                return res
            else:
                print("Error: Seleccion de camara no valida.")
        except ValueError:
            print("Error: entrada no valida, debe ser un entero.")            
    return res 
