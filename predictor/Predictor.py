"""
@author: Alberto Esteban Reyes Peralta
"""
import torch
import cv2
import math
import numpy as np
import configparser
import json
import PIL
import math
from PIL import Image
from torchvision import transforms 
from torchvision.utils import save_image
from os.path import exists
from os import mkdir
from pathlib import Path
from helpers import *
from super_gradients.common.object_names import Models
from super_gradients.training import models
import time
class Predictor:

    def __init__(self, camera, opt):        
        #Defincion del modelo y si el dispositivo con el que se trabajara
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        if (opt == 1):
            self.model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").to(self.device)
        else:
            self.model = models.get(Models.YOLO_NAS_S,num_classes=7,checkpoint_path = 'C:/Users/Usuario/Desktop/Project/ckpt_best.pth')
        #Seleccion de camara con la que se trabajara
        self.cap = cv2.VideoCapture(camera)
        #Si es video 
        #self.cap = cv2.VideoCapture('C:/Users/Usuario/Desktop/Project/Config/vids/inputVideo02.mp4')

        ##cargar los parametros de la camara
        #cameraMatrix, dist = pickle.load(open( os.path.join(self.constants['calibrationDirSaves'], 'calibration.pkl'), "rb" ))
        self.width = 640
        self.height= 360
        
        #newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (self.width,self.height), 1, (self.width,self.height))
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        with open('predictor.config') as json_file:
            self.dict_distances = json.load(json_file)

    def predict(self):    
        cv2.namedWindow("Identificacion y detección espacial de objetos", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Identificacion y detección espacial de objetos", 640, 360) 
        if not (self.cap.isOpened()):
            print("Error: No se detecto camara.")
            return -1
        names = []
        count = 0
        # Colors 
        GREEN = (0, 255, 0) 
        RED = (0, 0, 255) 
        WHITE = (255, 255, 255) 
        BLACK = (0, 0, 0) 
        frame_count = 0
        start_time = cv2.getTickCount()  # Get initial time
        while True:
            ret, frame = self.cap.read()   
            #ret, tmp = self.cap.read()
            #frame = cv.undistort(tmp, cameraMatrix, dist, None, newCameraMatrix)
            frame_count += 1
            if ret:    
                result = self.model.predict(frame, conf=0.70,fuse_model=False)#[0]
                #result = self.model.predict(frame, conf=0.60,fuse_model=False,skip_image_resizing=True)[0]
                if len(names)==0:
                    names=result.class_names
                bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
                confidences = result.prediction.confidence
                labels = result.prediction.labels.tolist()
                        
                for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                    bbox = np.array(bbox_xyxy)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_width_in_frame = x2 - x1
                    classname = int(cls)                    
                    class_name = names[classname]
                                                
                    if face_width_in_frame != class_name in self.dict_distances.keys():                        
                        # Encontrar la distancia
                        conf = math.ceil((confidence*100))/100
                        distance = self.distanceFinder(self.dict_distances[class_name]["focal_length"],self.dict_distances[class_name]["known_width"], face_width_in_frame) 
                        lbl_distance = f"{round(distance,3)} cm"
                        label = f'{class_name}-{lbl_distance}'
                    else:
                        label = f'{class_name}-{conf}'  

                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontScale = 2, thickness=1)[0]            
                    c2 = x1 + t_size[0], y1 - t_size[1] -3
                    cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1)
                    cv2.putText(frame, label, (x1, y1-2), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

              
                cv2.imshow("Identificacion y detección espacial de objetos", frame)        
                k = cv2.waitKey(1)
                if k%256 == 27: #ESC
                    break           
            else:
                break
        end_time = cv2.getTickCount()
        total_time = (end_time - start_time) / cv2.getTickFrequency()  # in seconds
        fps = frame_count / total_time
        print(f"Average FPS: {fps}")
        self.cap.release()
        cv2.destroyAllWindows()

    def predict_test(self):    
        cv2.namedWindow("Prueba de entrenamiento", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Prueba de entrenamiento", 640, 640) 
        if not (self.cap.isOpened()):
            print("Error: No se detecto camara.")
            return -1
        names = []
        while True:
            ret, frame = self.cap.read()    
            if ret:
                result = self.model.predict(frame, conf=0.55)[0]
                #result = self.model.predict(frame, conf=0.60,fuse_model=False,skip_image_resizing=True)[0]
                if len(names)==0:
                    names=result.class_names
                bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
                confidences = result.prediction.confidence
                labels = result.prediction.labels.tolist()
                        
                for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                    bbox = np.array(bbox_xyxy)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    face_width_in_frame = x2 - x1
                    classname = int(cls)                    
                    class_name = names[classname]                    
                                    
                    # Encontrar la distancia
                    label = f'{class_name}-{confidence}'
                    
                    # Colors 
                    GREEN = (0, 255, 0) 
                    RED = (0, 0, 255) 
                    WHITE = (255, 255, 255) 
                    BLACK = (0, 0, 0) 

                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontScale = 2, thickness=1)[0]            
                    c2 = x1 + t_size[0], y1 - t_size[1] -3
                    cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1)
                    cv2.putText(frame, label, (x1, y1-2), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
                cv2.imshow("Prueba de entrenamiento", frame)        
                k = cv2.waitKey(1)
                if k%256 == 27: #ESC
                    print("Saliendo")
                    return 1
            else:
                print("Error: Fallo en la captura de imagen")
                return -1
        self.cap.release()
        cv2.destroyAllWindows()
    
    #capturar imagen de referencia
    def getRefImage(self,save_path):
        cv2.namedWindow("Capturar imagen de referencia", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Capturar imagen de referencia", 640, 640) 
        if not (self.cap.isOpened()):
            print("Error: No se detecto camara.")
            self.cap.release()
            cv2.destroyAllWindows() 
            return -1                
        while True:
            ret, frame = self.cap.read()    
            if ret:                
                cv2.imshow("Capturar imagen de referencia", frame)        
                k = cv2.waitKey(1)
                if k%256 == 67 or k%256 == 99: #C o c                    
                    #save_path = self.dict_distances["matraz"]["ref_image_path"]
                    cv2.imwrite(save_path,frame)
                    #self.cap.release()
                    cv2.destroyAllWindows() 
                    return 1
                if k%256 == 27: #ESC                    
                    return 0
            else:
                print("Error: Fallo en la captura de imagen")
                self.cap.release()
                cv2.destroyAllWindows() 
                return -1
        
    def getWidth(self,Ref_image_path,class_int):
        print(Ref_image_path)
        result = self.model.predict(Ref_image_path, conf=0.50,fuse_model=False)[0]      
        print(result)  
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
                        
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):   
            print("here")         
            print(cls)
            if cls == class_int:
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = x2 - x1
                return w        
        return 0

    # Obtener la distancia focal
    def focalLengthFinder(self, measured_distance, real_width, width_in_rf_image): 
        focal_length = (width_in_rf_image * measured_distance) / real_width 
        return focal_length 

    # Estimacion de la distancia
    def distanceFinder(self, Focal_Length, real_face_width, face_width_in_frame): 
        distance = (real_face_width * Focal_Length)/face_width_in_frame 
        return distance 

    def getFocalDistances(self):
        change = False
        for key in self.dict_distances:            
            if not exists(self.dict_distances[key]["ref_image_path"]):   
                print("Capturar imagen para ",key)             
                self.getRefImage(self.dict_distances[key]["ref_image_path"])
                self.dict_distances[key]["width"] = self.getWidth(self.dict_distances[key]["ref_image_path"],self.dict_distances[key]["class"])
                self.dict_distances[key]["focal_length"] = self.focalLengthFinder(self.dict_distances[key]["known_distance"],self.dict_distances[key]["known_width"],self.dict_distances[key]["width"])
                change = True
            elif self.dict_distances[key]["width"] == 0:                
                self.dict_distances[key]["width"] = self.getWidth(self.dict_distances[key]["ref_image_path"],self.dict_distances[key]["class"])
                self.dict_distances[key]["focal_length"] = self.focalLengthFinder(self.dict_distances[key]["known_distance"],self.dict_distances[key]["known_width"],self.dict_distances[key]["width"])
                change = True
            elif self.dict_distances[key]["focal_length"] == 0:                
                self.dict_distances[key]["focal_length"] = self.focalLengthFinder(self.dict_distances[key]["known_distance"],self.dict_distances[key]["known_width"],self.dict_distances[key]["width"])
                change = True
        
        if change:
            print(self.dict_distances)
            with open("predictor.config", "w") as outfile: 
                json.dump(self.dict_distances, outfile)                



#Seleccion de camara con la que se trabajara
#cam = selectCamera()
cam = 0
oPredictor = Predictor(cam,2)
#oPredictor.getFocalDistances()
oPredictor.predict()

#oPredictor = Predictor(cam,2)
#oPredictor.predict()
#oPredictor.predict_test()