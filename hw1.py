import os
import sys
import copy
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from hw1_ui import Ui_MainWindow
import numpy as np
from matplotlib import pyplot as plt


class myMainWindow(QtWidgets.QMainWindow , Ui_MainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow , self).__init__()
        self.setupUi(self)
        self.onBindingUI()
        self.listOfImage = []
        self.imgL = None
        self.imgR = None
        self.folderPath = None

    def onBindingUI(self):
        self.pushButton.clicked.connect(self.loadFolder)
        self.pushButton_2.clicked.connect(self.loadImage_L)
        self.pushButton_3.clicked.connect(self.loadImage_R)
        self.pushButton_4.clicked.connect(self.findCorner)
        self.pushButton_5.clicked.connect(self.findIntrinsic)
        self.pushButton_6.clicked.connect(self.findExtrinsic)
        self.pushButton_7.clicked.connect(self.findDistortion)
        self.pushButton_8.clicked.connect(self.showResult)
        self.pushButton_9.clicked.connect(self.showWordsOnBoard)
        self.pushButton_10.clicked.connect(self.showWordsVertically)
        self.pushButton_11.clicked.connect(self.stereoDisparityMap)
    def loadFolder(self):
        self.folderPath = QtWidgets.QFileDialog.getExistingDirectory(self, 'folder')
        #path = './dataBase/calibration'
        for image in os.listdir(self.folderPath):
            img = cv2.imread(self.folderPath + '/' + image)
            self.listOfImage.append(img)

    def loadImage_L(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'folder')
        #path = './dataBase/disparity/imL.png'
        self.imgL = cv2.imread(path)
    def loadImage_R(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'folder')
        #path = './dataBase/disparity/imR.png'
        self.imgR = cv2.imread(path)

    def findCorner(self):
        for i, img in enumerate(self.listOfImage):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                cv2.namedWindow((str(i) + '.bmp' + 'Corner detection '), cv2.WINDOW_NORMAL)
                cv2.imshow((str(i) + '.bmp' + 'Corner detection '), img)
    def findIntrinsic(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)
        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        print('Intrinsic: \n' + str(mtx))
    def findExtrinsic(self):
        targetImgPath = self.folderPath + self.comboBox.currentText() + '.bmp'
        img = cv2.imread(targetImgPath)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)
        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)

        R = cv2.Rodrigues(np.array(rvecs[int(self.comboBox.currentText()) - 1]))
        R1 = R[0]
        t1 = tvecs[int(self.comboBox.currentText()) - 1]
        R2 = np.zeros([3, 4])
        for i in range(0, 3):
            for j in range(0, 3):
                R2[i, j] = R1[i, j]
        for i in range(0, 3):
            R2[i, 3] = t1[i]
        print('Extrinsic :\n' + str(R2))

    def findDistortion(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)
        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        print('Distortion : \n' + str(dist))

    def showResult(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            if ret == True:
                objpoints.append(objp)
                # cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imgpoints.append(corners)

        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        result = []
        for i ,img in enumerate(self.listOfImage):
            h, w = img.shape[:2]
            newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
            res = cv2.hconcat([img, dst])
            result.append(res)
            cv2.namedWindow(str(i) + '.bmp distortion', cv2.WINDOW_NORMAL)
            cv2.imshow(str(i) + '.bmp distortion', res)
    def showWordsOnBoard(self):
        text = self.textEdit.toPlainText()
        ArrayOfText = []
        fs = cv2.FileStorage('./Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        for ch in text:
            tmpMat = fs.getNode(ch).mat()
            ArrayOfText.append(tmpMat)


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        img_count = 0
        retlist = np.zeros([20])
        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            retlist[img_count] = ret
            img_count = img_count + 1
            if ret == True:
                objpoints.append(objp) #3D
                imgpoints.append(corners) #2D
        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        # AR
        count = 0  # count 5 picture
        for img_no in range(0, 20):
            if retlist[img_no] == 1:
                R = cv2.Rodrigues(np.array(rvecs[count]))
                R1 = R[0]
                t1 = tvecs[count]
                R2 = np.zeros([3, 4]) #R + T 外部參數矩陣
                for i in range(0, 3):
                    for j in range(0, 3):
                        R2[i, j] = R1[i, j]
                for i in range(0, 3):
                    R2[i, 3] = t1[i]

                #list = [np.array([[2.], [2.], [0.], [1.]]), np.array([[2.], [0.], [0.], [1.]]),
                #       np.array([[0.], [0.], [0.], [1.]]), np.array([[0.], [2.], [0.], [1.]]),
                #        np.array([[2.], [2.], [-2.], [1.]]), np.array([[2.], [0.], [-2.], [1.]]),
                #        np.array([[0.], [0.], [-2.], [1.]]), np.array([[0.], [2.], [-2.], [1.]])]
                list = []
                for i , obj in enumerate(ArrayOfText): #第i個字 #[3*2*3]
                    tmplist = [] #每個tmplist代表一個字
                    for j in obj: #[2*3]
                        for h in j: #[3.]
                            if(i < 3):
                                x = np.ones([4,1])
                                x[0] = h[0] + 7 - (i * 3)
                                x[1] = h[1] + 5
                                x[2] = h[2] + 0
                                tmplist.append(x)
                            else:
                                x = np.ones([4, 1])
                                x[0] = h[0] + 7 - ((i - 3) * 3)
                                x[1] = h[1] + 2
                                x[2] = h[2] + 0
                                tmplist.append(x)
                    list.append(tmplist)
                img1 = copy.deepcopy(self.listOfImage[img_no])

                for i , obj in enumerate(list):
                    for k,j in enumerate(obj):
                        a = np.dot(np.dot(mtx, R2), j) #(3,3) (3,4) list是要投影的圖像世界座標 mtx是內部參數矩陣
                        x = a[0] / a[2]
                        y = a[1] / a[2]
                        list[i][k] = (x, y) #image 座標
                print(len(list[0]))

                count = count + 1
                for i , text in enumerate(list):
                    for j in range(0 , len(text) ,2):
                        cv2.line(img1, list[i][j], list[i][j + 1], (0, 0, 255), 10)

                cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
                cv2.imshow('Img', img1)
                cv2.waitKey(1000)
            if count == 5:
                break

    def showWordsVertically(self):
        text = self.textEdit.toPlainText()
        ArrayOfText = []
        fs = cv2.FileStorage('./Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        for ch in text:
            tmpMat = fs.getNode(ch).mat()
            ArrayOfText.append(tmpMat)


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8 * 11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        img_count = 0
        retlist = np.zeros([20])
        for img in self.listOfImage:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            retlist[img_count] = ret
            img_count = img_count + 1
            if ret == True:
                objpoints.append(objp) #3D
                imgpoints.append(corners) #2D
        ret2, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w), None, None)
        # AR
        count = 0  # count 5 picture
        for img_no in range(0, 20):
            if retlist[img_no] == 1:
                R = cv2.Rodrigues(np.array(rvecs[count]))
                R1 = R[0]
                t1 = tvecs[count]
                R2 = np.zeros([3, 4]) #R + T 外部參數矩陣
                for i in range(0, 3):
                    for j in range(0, 3):
                        R2[i, j] = R1[i, j]
                for i in range(0, 3):
                    R2[i, 3] = t1[i]

                #list = [np.array([[2.], [2.], [0.], [1.]]), np.array([[2.], [0.], [0.], [1.]]),
                #       np.array([[0.], [0.], [0.], [1.]]), np.array([[0.], [2.], [0.], [1.]]),
                #        np.array([[2.], [2.], [-2.], [1.]]), np.array([[2.], [0.], [-2.], [1.]]),
                #        np.array([[0.], [0.], [-2.], [1.]]), np.array([[0.], [2.], [-2.], [1.]])]
                list = []
                for i , obj in enumerate(ArrayOfText): #第i個字 #[3*2*3]
                    tmplist = [] #每個tmplist代表一個字
                    for j in obj: #[2*3]
                        for h in j: #[3.]
                            if(i < 3):
                                x = np.ones([4,1])
                                x[0] = h[0] + 7 - (i * 3)
                                x[1] = h[1] + 5
                                x[2] = h[2] + 0
                                tmplist.append(x)
                            else:
                                x = np.ones([4, 1])
                                x[0] = h[0] + 7 - ((i - 3) * 3)
                                x[1] = h[1] + 2
                                x[2] = h[2] + 0
                                tmplist.append(x)
                    list.append(tmplist)
                img1 = copy.deepcopy(self.listOfImage[img_no])

                for i , obj in enumerate(list):
                    for k,j in enumerate(obj):
                        a = np.dot(np.dot(mtx, R2), j) #(3,3) (3,4) list是要投影的圖像世界座標 mtx是內部參數矩陣
                        x = a[0] / a[2]
                        y = a[1] / a[2]
                        list[i][k] = (x, y) #image 座標
                count = count + 1
                for i, text in enumerate(list):
                    for j in range(0, len(text), 2):
                        cv2.line(img1, list[i][j], list[i][j + 1], (0, 0, 255), 10)


                cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
                cv2.imshow('Img', img1)
                cv2.waitKey(1000)
            if count == 5:
                break
    def L_clicked_event(self , event , x , y ,flags , param):
        if event == cv2.EVENT_LBUTTONDOWN:
            L_img = cv2.circle(copy.deepcopy(self.imgL), (x, y), 10, (255, 0, 0), -1)
            cv2.imshow('imgL', L_img)
            disp = param[0]
            d = disp[y][x] // 16
            print(d)
            if(d == 0):
                cv2.imshow('imgR', self.imgR)
                return
            R_img = cv2.circle(copy.deepcopy(self.imgR), (x - d , y), 10, (0, 0, 255), -1)
            cv2.imshow('imgR' , R_img)


    def stereoDisparityMap(self):
        cImgL = copy.deepcopy(self.imgL)
        cImgL = cv2.cvtColor(cImgL , cv2.COLOR_BGR2GRAY)
        cImgR = copy.deepcopy(self.imgR)
        cImgR = cv2.cvtColor(cImgR,cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparityGray = stereo.compute(cImgL, cImgR)
        disparity = cv2.normalize(disparityGray, disparityGray, alpha=255,beta=0, norm_type=cv2.NORM_MINMAX)
        disparity = np.uint8(disparity)
        cv2.namedWindow('Stereo Disparity Map', cv2.WINDOW_NORMAL)
        cv2.imshow('Stereo Disparity Map', disparity)


        stereoRGB = cv2.StereoSGBM_create(numDisparities=256,blockSize=25)
        disparityRGB = stereoRGB.compute(self.imgL , self.imgR)



        cv2.namedWindow('imgL' ,cv2.WINDOW_NORMAL)
        cv2.imshow('imgL', self.imgL)
        cv2.resizeWindow('imgL', 500, 500)

        cv2.namedWindow('imgR' , cv2.WINDOW_NORMAL)
        cv2.imshow('imgR', self.imgR)
        cv2.resizeWindow('imgR' , 500 , 500)

        param = [disparityRGB]
        cv2.setMouseCallback('imgL', self.L_clicked_event , param)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())