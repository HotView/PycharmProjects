#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
def Steger(img):
    gray_origin = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("gray_origin",0)
    cv2.imshow("gray_origin",gray_origin)
    gray = cv2.GaussianBlur(gray_origin,(5,5),0)
    Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
    # Hessian矩阵
    row = img.shape[0]
    col = img.shape[1]
    CenterPoint = []
    for i in range(col):
        for j in range(row):
            if gray_origin[j,i]>200:
                hessian = np.zeros((2,2),np.float32)
                hessian[0,0] = Ixx[j,i]
                hessian[0,1] = Ixy[j,i]
                hessian[1,0] = Iyx[j,i]
                hessian[1,1] = Iyy[j,i]
                ret,eigenVal,eigenVec= cv2.eigen(hessian)
                nx,ny,fmaxD = 0.0,0.,0.
                if ret:
                    #print(eigenVal.shape,eigenVec.shape)
                    if np.abs(eigenVal[0,0]>=eigenVal[1,0]):
                        nx = eigenVec[0,0]
                        ny = eigenVec[0,1]
                        famxD = eigenVal[0,0]
                    else:
                        nx = eigenVec[1, 0]
                        ny = eigenVec[1, 1]
                        famxD = eigenVal[1, 0]
                    t = -(nx * Ix[j, i] + ny * Iy[j, i]) / (
                                nx * nx * Ixx[j, i] + 2 * nx * ny * Ixy[j, i] + ny * ny * Iyy[j, i])
                    if np.abs(t*nx)<=0.5 and np.abs(t*ny)<=0.5:
                        CenterPoint.append([i,j])
    cv2.namedWindow("Steger_origin",0)
    cv2.imshow("Steger_origin",img)
    for point in CenterPoint:
        #cv2.circle(img,(point[0],point[1]),1,(0,255,0))
        img[point[1],point[0]] = (0,255,0)
    cv2.namedWindow("res", 0)
    cv2.imshow("res",img)
def StegerLine(img):
    gray_origin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_origin, (5, 5), 0)
    Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0,)
    Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
    Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
    Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
    Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
    # Hessian矩阵
    row = gray_origin.shape[0]
    col = gray_origin.shape[1]
    CenterPoint = []
    for i in range(col):
        for j in range(row):
            if gray_origin[j, i] > 200:
                hessian = np.zeros((2, 2), np.float32)
                hessian[0, 0] = Ixx[j, i]
                hessian[0, 1] = Ixy[j, i]
                hessian[1, 0] = Iyx[j, i]
                hessian[1, 1] = Iyy[j, i]
                ret, eigenVal, eigenVec = cv2.eigen(hessian)
                lambda1 = 0.
                lambda2 = 0.
                nx, ny, fmaxD = 0.0, 0., 0.
                if ret:
                    # print(eigenVal.shape,eigenVec.shape)
                    if np.abs(eigenVal[0, 0]) >= np.abs(eigenVal[1, 0]):
                        lambda1 = eigenVal[1,0]
                        lambda2 = eigenVal[0,0]
                        nx = eigenVec[0, 0]
                        ny = eigenVec[0, 1]
                        famxD = eigenVal[0, 0]
                    else:
                        lambda1 = eigenVal[0, 0]
                        lambda2 = eigenVal[1, 0]
                        nx = eigenVec[1, 0]
                        ny = eigenVec[1, 1]
                        famxD = eigenVal[1, 0]
                    #if lambda1<15 and lambda2<-50:
                    t = -(nx * Ix[j, i] + ny * Iy[j, i]) / (
                                nx * nx * Ixx[j, i] + 2 * nx * ny * Ixy[j, i] + ny * ny * Iyy[j, i])
                    if np.abs(t * nx) <= 0.5 and np.abs(t * ny) <= 0.5:
                            #CenterPoint.append([i, j])
                        CenterPoint.append([i, j])
    cv2.namedWindow("Steger_origin", 0)
    new_img = np.zeros((row,col),np.uint8)
    cv2.imshow("Steger_origin", gray_origin)
    for point in CenterPoint:
        cv2.circle(img, (point[0], point[1]), 1, 255)
    cv2.namedWindow("res", 0)
    cv2.imshow("res", img)
def test(filename):
    img = cv2.imread(filename)
    start = cv2.getTickCount()
    Steger(img)
    print("spend",(cv2.getTickCount()-start)/cv2.getTickFrequency())
    cv2.waitKey(0)
def test02(filename):
    laser_bin = cv2.imread(filename)
    start = cv2.getTickCount()
    StegerLine(laser_bin)
    print("spend", (cv2.getTickCount() - start) / cv2.getTickFrequency())
    cv2.waitKey(0)
if __name__ == '__main__':
    fn = "center-v.jpg"
    test02(fn)








