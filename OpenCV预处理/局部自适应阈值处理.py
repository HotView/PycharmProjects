# 目标区域为亮区，即灰度值最大的地方为感兴趣区
import cv2
import numpy as np
from OpenCV边缘提取与ROI.GetMaxRegion import getmaxRect

def Nothing(val):
    img2 = img.copy()
    size = cv2.getTrackbarPos("size","gray")
    param = cv2.getTrackbarPos("param","gray")
    thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,2*size+1, -param)
    x,y,w,h = getmaxRect(thresh)
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Rect_Res", img2)
    cv2.imshow("thresh", thresh)
img = cv2.imread("image/laser_test01.jpg")
img = cv2.GaussianBlur(img,(5,5),1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("thresh")
cv2.namedWindow("gray")
cv2.createTrackbar("size","gray",0,100,Nothing)
cv2.createTrackbar("param","gray",0,100,Nothing)
cv2.imshow("gray",gray)
cv2.waitKey(0)