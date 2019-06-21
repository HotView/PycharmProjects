#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np
def Extreme(img):
    row,col,chanel = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    points= []
    newimage = np.zeros((row,col),np.uint8)
    for i in range(col):
        Pmax = np.max(gray[:,i])
        Prow = np.argmax(gray[:,i])
        points.append([Prow,i])
    for p in points:
        newimage[p[0],p[1]] = 255
        img[p[0],p[1],:] = [0,255,0]
    cv2.namedWindow("origin",0)
    cv2.namedWindow("centerLine",0)
    cv2.imshow("origin",img)
    cv2.imshow("centerLine",newimage)
img = cv2.imread("laser-v.jpg")
Extreme(img)
cv2.waitKey(0)