#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np
def getLines(gray):
    minLineLength = 15
    maxLineGap = 5
    lines = cv2.HoughLinesP(gray, 1.0, np.pi/180, 10, minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines
def drawLine(lines,img):
    for line in lines:
        line = line[0]
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), 255)

def Gravity(img):
    row,col,chanel = img.shape
    lineimage = np.zeros((row,col),dtype=np.uint8)
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
    lines = getLines(newimage)
    drawLine(lines,lineimage)
    cv2.imshow("origin",img)
    cv2.imshow("lineimage",lineimage)
    cv2.imshow("centerLine",newimage)
def GravityPlus(img,thresh):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = np.zeros((col,2))
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        Pmax = np.max(gray[:, i])
        #Pmin  = np.min(gray[:, i])
        if Pmax<thresh:
            continue
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
            points[i]=[Prow[0],i]
    print(points)
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
def GravityPlusK(img,thresh,k):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = np.zeros((col,2))
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        posMax = np.argmax(gray[:, i])
        Pmax = gray[posMax, i]
        #Pmin  = np.min(gray[:, i])
        if Pmax<thresh:
            continue
        sumPix = 0
        sumVal = 0
        for index in range(-k,k):
            sumPix+=gray[posMax+index,i]*(posMax+index)
            sumVal+=gray[posMax+index,i]
        valCenter =sumPix/sumVal
        points[i]=[valCenter,i]
    print(points)
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
img = cv2.imread("image/laser-v.jpg")
GravityPlus(img,100,)
cv2.waitKey(0)