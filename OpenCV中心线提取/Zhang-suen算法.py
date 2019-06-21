#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def ROI(img):
    indexXY = np.argwhere(img > 0)
    minxy = np.min(indexXY, axis=0)
    maxxy = np.max(indexXY, axis=0)
    return minxy,maxxy
def neighbours(x,y,img):
    i = img
    x1,y1,x_1, y_1 = x+1, y-1, x-1, y+1
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9
def transitions(neighbours):
    n = neighbours + neighbours[0:1]  # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))
def ZhangSuenPlus(image):
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x,y))
        for x, y in changing1: image[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x,y))
        for x, y in changing2: image[y][x] = 0
        #print changing1
        #print changing2
    flags = image>0
    image[flags] = 255
    #cv2.imshow("res",image)
    return image
def ZhangSuenPlus02(image):
    """
    # 运行时间12.135秒
    :param image:
    :return:
    """
    indexXY = np.argwhere(image>0)
    minxy = np.min(indexXY,axis=0)
    maxxy = np.max(indexXY,axis=0)
    roi = image[minxy[0]-1:maxxy[0]+2,minxy[1]-1:maxxy[1]+2]
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, roi)
                    if (    # (Condition 0)
                        P4 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P6 == 0 and   # Condition 3
                        transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing1.append((x,y))
        for x, y in changing1: roi[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, roi)
                    if (   # (Condition 0)
                        P2 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing2.append((x,y))
        for x, y in changing2: roi[y][x] = 0
        #print changing1
        #print changing2
    flags = roi>0
    roi[flags] = 255
    #cv2.imshow("res",image)
    return image
def ZhangSuenPlus03(image):
    """
    # 运行时间9秒
    :param image:
    :return:
    """
    indexXY = np.argwhere(image>0)
    minxy = np.min(indexXY,axis=0)
    maxxy = np.max(indexXY,axis=0)
    roi = image[minxy[0]-1:maxxy[0]+2,minxy[1]-1:maxxy[1]+2]
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        indexXY = np.argwhere(roi>0)
        minxy = np.min(indexXY, axis=0)
        maxxy = np.max(indexXY, axis=0)
        roi = roi[minxy[0] - 1:maxxy[0] + 2, minxy[1] - 1:maxxy[1] + 2]
        # Step 1
        changing1 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, roi)
                    if (    # (Condition 0)
                        P4 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P6 == 0 and   # Condition 3
                        transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing1.append((x,y))
        for x, y in changing1: roi[y][x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(roi) - 1):
            for x in range(1, len(roi[0]) - 1):
                if roi[y][x] == 1:
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, roi)
                    if (   # (Condition 0)
                        P2 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing2.append((x,y))
        for x, y in changing2: roi[y][x] = 0
        #print changing1
        #print changing2
    flags = roi>0
    roi[flags] = 255
    #cv2.imshow("res",image)
    return image

def ZhangSuen_Bad(img):
    copyMat = img.copy()
    k = 0
    row,col= img.shape
    row = row-1
    col = col-1
    while(True):
        k= k+1
        stop= False
        # step1
        for i in range(1,row):
            for j in range(1,col):
                if img[i,j]>0:
                    print(">0")
                    p1 = 1 if img[i,j]>0 else 0
                    p2 = 1 if img[i-1,j]>0 else 0
                    p3 = 1 if img[i-1,j+1]>0 else 0
                    p4 = 1 if img[i, j+1] > 0 else 0
                    p5 = 1 if img[i+1, j+1] > 0 else 0
                    p6 = 1 if img[i+1, j] > 0 else 0
                    p7 = 1 if img[i+1, j-1] > 0 else 0
                    p8 = 1 if img[i,j-1] > 0 else 0
                    p9 = 1 if img[i-1, j-1] > 0 else 0
                    np1 = p2+p3+p4+p5+p6+p7+p8+p9
                    sp2 = 1 if (p2 == 0 and p3 == 1) else 0
                    sp3 = 1 if (p3 == 0 and p4 == 1) else 0
                    sp4 = 1 if (p4 == 0 and p5 == 1) else 0
                    sp5 = 1 if (p5 == 0 and p6 == 1) else 0
                    sp6 = 1 if (p6 == 0 and p7 == 1) else 0
                    sp7 = 1 if (p7 == 0 and p8 == 1) else 0
                    sp8 = 1 if (p8 == 0 and p9 == 1) else 0
                    sp9 = 1 if (p9 == 0 and p2 == 1) else 0
                    sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9
                    if np1>=2 and np1<=6 and sp1==1 and(p2*p4*p6)==0 and (p4*p6*p8)==0:
                        stop = True
                        copyMat[i,j] = 0
                        print("success")
        img = copyMat.copy()
        # step2
        for i in range(1,row):
            for j in range(1,col):
                if img[i,j]>0:
                    print(">>")
                    p2 = 1 if img[i - 1, j] > 0 else 0
                    p3 = 1 if img[i - 1, j + 1] > 0 else 0
                    p4 = 1 if img[i, j + 1] > 0 else 0
                    p5 = 1 if img[i + 1, j + 1] > 0 else 0
                    p6 = 1 if img[i + 1, j] > 0 else 0
                    p7 = 1 if img[i + 1, j - 1] > 0 else 0
                    p8 = 1 if img[i, j - 1] > 0 else 0
                    p9 = 1 if img[i - 1, j - 1] > 0 else 0
                    np1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    sp2 = 1 if (p2 == 0 and p3 == 1) else 0
                    sp3 = 1 if (p3 == 0 and p4 == 1) else 0
                    sp4 = 1 if (p4 == 0 and p5 == 1) else 0
                    sp5 = 1 if (p5 == 0 and p6 == 1) else 0
                    sp6 = 1 if (p6 == 0 and p7 == 1) else 0
                    sp7 = 1 if (p7 == 0 and p8 == 1) else 0
                    sp8 = 1 if (p8 == 0 and p9 == 1) else 0
                    sp9 = 1 if (p9 == 0 and p2 == 1) else 0
                    sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9
                    if np1 >= 2 and np1 <= 6 and sp1 == 1 and (p2*p4*p8) == 0 and (p2*p6*p8) == 0:
                        stop = True
                        copyMat[i,j] = 0
                        print("success")
        img = copyMat.copy()
        if(not stop):
            break
    resImg = copyMat.copy()
    flags = resImg>0
    resImg[flags] = 255
    #print(k)
    # cv2.imshow("res",resImg)
    return resImg

def ZhangSuen(img):
    #indexXY = np.argwhere(img>0)
    #minxy = np.min(indexXY,axis=0)
    #maxxy = np.max(indexXY,axis=0)
    #roi = img[minxy[0]-3:maxxy[0]+4,minxy[1]-3:maxxy[1]+4]
    #flags = roi>0
    #roi[flags] = 255
    #cv2.imshow("roi",roi)
    #print(roi.shape)
    roi = img
    k = 0
    row,col= roi.shape
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        changing1 = []
        for i in range(1,row-1):
            for j in range(1,col-1):
                if roi[i,j]==1:
                    p2 = roi[i-1,j]
                    p3 = roi[i-1,j+1]
                    p4 = roi[i, j+1]
                    p5 = roi[i+1, j+1]
                    p6 = roi[i+1, j]
                    p7 = roi[i+1, j-1]
                    p8 = roi[i,j-1]
                    p9 = roi[i-1, j-1]
                    np1 = p2+p3+p4+p5+p6+p7+p8+p9
                    sp2 = 1 if (p2,p3)==(0,1) else 0
                    sp3 = 1 if (p3,p4)==(0,1) else 0
                    sp4 = 1 if (p4,p5)==(0,1) else 0
                    sp5 = 1 if (p5,p6)==(0,1) else 0
                    sp6 = 1 if (p6,p7)==(0,1) else 0
                    sp7 = 1 if (p7,p8)==(0,1) else 0
                    sp8 = 1 if (p8,p9)==(0,1) else 0
                    sp9 = 1 if (p9,p2)==(0,1) else 0
                    sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9
                    if 2<=np1<=6 and sp1==1 and(p2*p4*p6)==0 and (p4*p6*p8)==0:
                        changing1.append([i,j])
        for x,y in changing1:roi[x,y] = 0
        # step2
        changing2 = []
        for i in range(1,row-1):
            for j in range(1,col-1):
                if roi[i,j]==1:
                    p2 = roi[i - 1, j]
                    p3 = roi[i - 1, j + 1]
                    p4 = roi[i, j + 1]
                    p5 = roi[i + 1, j + 1]
                    p6 = roi[i + 1, j]
                    p7 = roi[i + 1, j - 1]
                    p8 = roi[i, j - 1]
                    p9 = roi[i - 1, j - 1]
                    np1 = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                    sp2 = 1 if (p2, p3) == (0, 1) else 0
                    sp3 = 1 if (p3, p4) == (0, 1) else 0
                    sp4 = 1 if (p4, p5) == (0, 1) else 0
                    sp5 = 1 if (p5, p6) == (0, 1) else 0
                    sp6 = 1 if (p6, p7) == (0, 1) else 0
                    sp7 = 1 if (p7, p8) == (0, 1) else 0
                    sp8 = 1 if (p8, p9) == (0, 1) else 0
                    sp9 = 1 if (p9, p2) == (0, 1) else 0
                    sp1 = sp2 + sp3 + sp4 + sp5 + sp6 + sp7 + sp8 + sp9
                    if 2<=np1<= 6 and sp1 == 1 and (p2*p4*p8) == 0 and (p2*p6*p8) == 0:
                        #roi[i,j] = 0
                        changing2.append([i,j])
                        #print("success")
        for x,y in changing2:roi[x,y] = 0
    flags = roi>0
    roi[flags] = 255
    return roi
img =  cv2.imread("laser_bin.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Origin",gray)
ret,thresh = cv2.threshold(gray,125,1,cv2.THRESH_BINARY)

time1 = time.time()
#res= ZhangSuen(thresh)
time2 = time.time()
#resP= ZhangSuenPlus(thresh)
resP2 = ZhangSuenPlus02(thresh)
#resroi = ZhangSuen_Bad(thresh)
time3 =time.time()
cv2.namedWindow("thresh",0)
cv2.namedWindow("resroi",0)
cv2.imshow("thresh",thresh)
cv2.imshow("resroi",resP2)
print("normal time spend:",time2-time1)
print("plus time spend:",time3-time2)
cv2.waitKey(0)
def Draw():
    plt.figure()
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.title("origin image")
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(res,"gray")
    plt.title("res image")
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(resP,"gray")
    plt.title("resP image")
    plt.axis("off")
    plt.show()
#Draw()








