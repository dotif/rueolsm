import albumentations as A
import cv2
import os
import pybboxes
import json
import pybboxes as pbx
from pathlib import Path

class DataAugmentor:

    def __init__(self, const):
        self.constants = const

    def getInputData(self, imgFile):
        fileNameExt = Path(imgFile).name
        fileName = Path(fileNameExt).stem
     
        augFileName = f"{fileName}_{self.constants['transformedFileName']}"
        image = cv2.imread(os.path.join(self.constants['inpImgPth'],fileNameExt))
        labPth = os.path.join(self.constants['inpLabPth'], f"{fileName}.txt")
        txt = self.constants['CLASSES']
        gtBboxes = self.getBboxesList(labPth,txt.split(',')) 
        return image, gtBboxes, augFileName
    
    def getBboxesList(self, inpLabPth, classes):
        yoloStrLabels = open(inpLabPth,'r').read()
        
        if not yoloStrLabels:
            print('No object')
            return []
        
        lines = [line.strip() for line in yoloStrLabels.split('\n') if line.strip()]
        albumBbList = self.getAlbumBbLists('\n'.join(lines), classes) if len(lines) > 1 else [self.getAlbumBbList('\n'.join(lines), classes)]

        return albumBbList
    
    def getAlbumBbLists(self, yoloStrLabels, classes):
        albumBbLists = []
        yoloListLabels = yoloStrLabels.split('\n')
        for yoloStrLabel in yoloListLabels:
            if yoloStrLabel:
                albumBbList = self.getAlbumBbList(yoloStrLabel, classes)
                albumBbLists.append(albumBbList)
        return albumBbLists
    
    def getAlbumBbList(self, yoloBbox, classNames):
        strBboxList = yoloBbox.split()
        classNumber = int(strBboxList[0])
        className = classNames[classNumber]
        bboxValues = list(map(float, strBboxList[1:]))
        albumBB = bboxValues + [className]
        return albumBB
    
    def singleObjBbYoloConversion(self, transformedBboxes, classNames):
        if transformedBboxes:
            classNum = classNames.index(transformedBboxes[-1])
            bboxes = list(transformedBboxes)[:-1]
            bboxes.insert(0, classNum)
        else:
            bboxes = []
        return bboxes
    
    def multiObjBbYoloConversion(self, augLabs, classNames):
        yoloLabels = [self.singleObjBbYoloConversion(augLab, classNames) for augLab in augLabs]
        return yoloLabels
    
    def saveAugLab(self, transformedBboxes, labPth, labName):
        labOutPath = os.path.join(labPth, labName)
        with open(labOutPath, 'w') as output:
            for bbox in transformedBboxes:
                updateBbox = str(bbox).replace(',', '').replace('[', '').replace(']', '')
                output.write(updateBbox+'\n')

    def saveAugImage(self, transformedImage, outImgPth, imgName):
        outImgPath = os.path.join(outImgPth, imgName) 
        cv2.imwrite(outImgPath, transformedImage)

    def drawYolo(self, image, labels, fileName):
        H, W = image.shape[:2]
        for label in labels:
            yoloNormalized = label[1:]
            boxVoc = pbx.convert_bbox(tuple(yoloNormalized), from_type="yolo", to_type="voc", image_size=(W, H))
            cv2.rectangle(image, (boxVoc[0], boxVoc[1]), (boxVoc[2], boxVoc[3]) , (0, 0, 255), 1)
        cv2.imwrite(f"{self.constants['bbImage']}/{fileName}.png", image)

    def hasNegativeElement(self, lst):
        return any(x < 0 for x in lst)
    
    def getAugmentedResults(self, image, bboxes):
        
        h, w, channels = image.shape
        nh = 0
        nw = 0
        if(h==w and h<=640):
            nh = h
            nw = w
        else:
            nh = 360
            nw = 640
        

        transforms = []
        transformed = []

        #Imagen original
        transform01 = A.Compose([
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform01)
          
        #Recorte aleatorio
        if(nh!=nw):
            transform02 = A.Compose([
                A.BBoxSafeRandomCrop(),
                A.Rotate(limit=15,p=0.7,always_apply=True),
                A.Resize(height=nh,width=nw,always_apply=True)
            ],bbox_params=A.BboxParams(format='yolo'))
            transforms.append(transform02)
        
        #Volteo horizontal
        transform03 = A.Compose([
            A.HorizontalFlip(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform03)

        #Volteo vertical
        transform04 = A.Compose([
            A.VerticalFlip(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform04)

        #Escala de grises
        transform05 = A.Compose([
            A.ToGray(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform05)

        #Desenfoque aleatorio
        transform06 = A.Compose([
            A.Blur(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform06)

        #Ruido aleatorio 
        transform07 = A.Compose([
            A.GaussNoise(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform07)

        #Color aleatorio
        transform08 = A.Compose([
            A.RGBShift(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform08)

        #Invertir color
        transform09 = A.Compose([
            A.InvertImg(always_apply=True),
            A.Rotate(limit=15,p=0.7,always_apply=True),
            A.Resize(height=nh,width=nw,always_apply=True)
        ],bbox_params=A.BboxParams(format='yolo'))
        transforms.append(transform09)

        for transform in transforms:
            temp = transform(image=image, bboxes=bboxes)
            transformed.append(temp)
               
        return transformed
    
    def hasNegativeElement(self, matrix):
        return any(element < 0 for row in matrix for element in row)
    
    def saveAugmentation(self, transImage, transBboxes, transFileName):
        totObjs = len(transBboxes)
        if totObjs:
            transBboxes = self.multiObjBbYoloConversion(transBboxes, self.constants['CLASSES'].split(',')) if totObjs > 1 else [self.singleObjBbYoloConversion(transBboxes[0], self.constants['CLASSES'].split(','))]
            if not self.hasNegativeElement(transBboxes):
                self.saveAugLab(transBboxes, self.constants['outLabPth'], transFileName + '.txt')
                self.saveAugImage(transImage, self.constants['outImgPth'], transFileName + '.png')
                #self.drawYolo(transImage, transBboxes, transFileName)
            else:
                print("Se encontro un elemento negativo en BBox transformado...")
        else:
            print("El archivo de etiquetas esta vacio...")