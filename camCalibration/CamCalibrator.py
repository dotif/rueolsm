import numpy as np
import cv2 as cv
import glob
import os 
import pickle
import matplotlib.pyplot as plt 
import json

class CamCalibrator:
    def __init__(self,const,showPics=False,camIndex=0):
        self.constants = const
        self.showPics=showPics
        self.camIndex = camIndex

    
    def getImages(self):
        cap = cv.VideoCapture(self.camIndex)
        num = 0
        while cap.isOpened():
            sucess, img = cap.read()
            k = cv.waitKey(5)
            if k == 27:
                break
            elif k == ord('c') or k == ord('C'):
                name = os.path.join(self.constants['calibrationDir'],'img'+str(num)+'.png')
                cv.imwrite(name, img)
                print(f'Imagen {num} guradada.')
                num += 1
            cv.imshow('Imagen capturada', img)      
        cap.release()
        cv.destroyAllWindows()

    def calibrate(self):
        
        chessboardSize = (9,6)
        frameSize = (1920,1080)
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        size_of_chessboard_squares_mm = 25
        objp = objp * size_of_chessboard_squares_mm

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob(os.path.join(self.constants['calibrationDir'], '*.png'))

        for image in images:

            img = cv.imread(image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            if ret == True:

                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)

                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(100)
        cv.destroyAllWindows()

        ############## CALIBRATION #######################################################

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

        # Save
        pickle.dump((cameraMatrix, dist), open(os.path.join(self.constants['calibrationDirSaves'], 'calibration.pkl'), "wb" ), protocol = 2)
        pickle.dump(cameraMatrix, open( os.path.join(self.constants['calibrationDirSaves'], 'cameraMatrix.pkl'), "wb" ), protocol = 2)
        pickle.dump(dist, open(os.path.join(self.constants['calibrationDirSaves'], 'dist.pkl'), "wb" ), protocol = 2)

    def undistort(self, img):
        # Read in the saved objpoints and imgpoints
        cameraMatrix, dist = pickle.load(open( os.path.join(self.constants['calibrationDirSaves'], 'calibration.pkl'), "rb" ))
        #dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )

        # Read in an image
        h,  w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

        # Undistort
        undistorted = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        undistorted = cv.resize(undistorted,(640,480))
        cv.imshow("Original Image", img)
        name = os.path.join(self.constants['calibrationDirSaves'],'og_img.png')
        cv.imwrite(name, img)
        cv.imshow("Undistorted Image", undistorted)
        name = os.path.join(self.constants['calibrationDirSaves'],'un_img.png')
        cv.imwrite(name, undistorted)
        cv.waitKey(0)

        cv.destroyAllWindows()


with open('camCalibrator.config') as jsonFile:
        constants = json.load(jsonFile)
oCamCal = CamCalibrator(constants)
#oCamCal.getImages()
#oCamCal.calibrate()
img = cv.imread('C:/Users/Usuario/Documents/TesisDataAugmentation/SistemaDUE/calibration/test.png')#C:/Users/Usuario/Documents/TesisDataAugmentation/SistemaDUE/calibration/input-cal/img10.png')
oCamCal.undistort(img)