import cv2
import numpy as np
from OpenCV预处理.Morphology import *

def Active(pos):
    Gk = cv2.getTrackbarPos("Gaussion","control")
    thresh = cv2.getTrackbarPos("Thresh","control")
    Mk = cv2.getTrackbarPos("Medsize","control")
    k = cv2.getTrackbarPos("MediaK","control")
    flags_dila = cv2.getTrackbarPos("Dilation","control")
    flags_erod = cv2.getTrackbarPos("Erode", "control")
    flags_open = cv2.getTrackbarPos("Open", "control")
    flags_close = cv2.getTrackbarPos("Close", "control")
    ksize = 2*Gk+1
    ksize2 = 2*Mk+1
    print(ksize2)
    res = cv2.GaussianBlur(gray,(ksize,ksize), 1)
    ret,res = cv2.threshold(res,thresh,255,cv2.THRESH_BINARY_INV)
    for i in range(k):
        res = cv2.medianBlur(res,ksize2)
    if flags_dila:
        res = Dilation(res )
    if flags_erod:
        print("Erode")
        res = Erode(res)
    if flags_open:
        res = Opening(res)
    if flags_close:
        res = Closig(res)

    cv2.imshow("res",res)

filename = "hand01.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("control",0)
cv2.createTrackbar("Gaussion","control",1,10,Active)
cv2.createTrackbar("Thresh","control",1,255,Active)
cv2.createTrackbar("Medsize","control",1,5,Active)
cv2.createTrackbar("MediaK","control",1,5,Active)
cv2.createTrackbar("Dilation","control",0,1,Active)
cv2.createTrackbar("Erode","control",0,1,Active)
cv2.createTrackbar("Open","control",0,1,Active)
cv2.createTrackbar("Close","control",0,1,Active)
cv2.imshow("origin",img)
cv2.waitKey(0)