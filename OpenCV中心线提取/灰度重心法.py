#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np


def Gravity(img):
    row,col,chanel = img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    points= []
    newimage = np.zeros((row,col),np.uint8)
    for i in range(col):
        Pmax = np.max(gray[:,i])
        Prow = np.argmax(gray[:,i])
        #print(Prow)
        points.append([Prow,i])
    for p in points:
        #print(p)
        newimage[p[0],p[1]] = 255
        img[p[0],p[1],:] = [0,255,0]
    cv2.namedWindow("origin",0)
    cv2.namedWindow("centerLine",0)
    cv2.imshow("origin",img)
    cv2.imshow("centerLine",newimage)
def GravityPlus(img):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = []
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        Pmax = np.max(gray[:, i])
        Pmin  = np.min(gray[:, i])
        if Pmax==Pmin:
            continue
        #print("Pmax",Pmax)
        pos = np.argwhere(gray[:,i]>=(Pmax-5))
        #print("pos",pos)
        length = len(pos)
        sum_top,sum_down =0.0,0.0
        if pos[-1]-pos[0]==length-1:
            #print("good cols",i)
            for p in pos:
                sum_top += p*gray[p,i]
                sum_down+=gray[p,i]
            Prow = sum_top/sum_down
            points.append([Prow[0],i])
    for p in points:
        #print(p)
        pr,pc = map(int,p)
        newimage[pr,pc] = 255
        img[pr,pc,:] = [0,255,0]
    cv2.namedWindow("Plus_origin",0)
    cv2.namedWindow("Plus_centerLine",0)
    cv2.imshow("Plus_origin",img)
    cv2.imshow("Plus_centerLine",newimage)
    return points
img = cv2.imread("laser-v.jpg")
Gravity(img)
cv2.waitKey(0)