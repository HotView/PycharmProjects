#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Auther: Liu
import cv2
import numpy as np
import  matplotlib.pyplot as plt
from OpenCV预处理.Morphology import *

def Drawing():
    plt.figure()
    plt.subplot(231)
    plt.imshow(thresh,'gray')
    plt.title("thresh")
    plt.axis("off")
    plt.subplot(232)
    plt.imshow(media,'gray')
    plt.title("media")
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(open,'gray')
    plt.title("open")
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(close,'gray')
    plt.title("close")
    plt.axis("off")
    plt.subplot(235)
    plt.imshow(dilation,'gray')
    plt.title("dilation")
    plt.axis("off")
    plt.subplot(236)
    plt.imshow(media3,'gray')
    plt.title("media2")
    plt.axis("off")
    plt.show()
img = cv2.imread('laser02.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threval = 125
ret,thresh = cv2.threshold(gray,threval,255,cv2.THRESH_BINARY_INV)
media = cv2.medianBlur(thresh,3)
media2 = cv2.medianBlur(media,3)
media3 = cv2.medianBlur(media2,5)
dilation = Dilation(media3)
cv2.imshow("01",dilation)
#cv2.imwrite("laser_bin.jpg",dilation)
open = Opening(media)
close = Closig(open)
dilation = Dilation(open)
Drawing()




