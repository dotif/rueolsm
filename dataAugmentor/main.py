from DataAugmentor import *
import json
import glob
from pathlib import Path

def runDataAugmentor():
    #Abrir archivo de configuración
    with open('dataAugmentor.config') as jsonFile:
        constants = json.load(jsonFile)
    
    #Obtener las extensiones permitidas
    extensionsAllowed = constants['extensionsAllowed'].split(',')
    imgs = []
    for ext in extensionsAllowed:
        tempRes = glob.glob(os.path.join(constants['inpImgPth'], ext))
        imgs += tempRes
    
    oDataAugmentor = DataAugmentor(constants)
    #Obtener el aumento de datos de cada imagen
    for imgNum, imgFile in enumerate(imgs): 
        print(f"Procesando imagen [{imgNum+1}]: {Path(imgFile).name}")
        image, gtBboxes, augFileName = oDataAugmentor.getInputData(imgFile)
        transformedImages = oDataAugmentor.getAugmentedResults(image,gtBboxes)
        for i,transformedImage in enumerate(transformedImages):
            augImg = transformedImage['image']
            augLabel = transformedImage['bboxes']
            if len(augImg) and len(augLabel):
                oDataAugmentor.saveAugmentation(augImg, augLabel, f"{augFileName}_{i+1}")
        
if __name__ == "__main__":
      runDataAugmentor()