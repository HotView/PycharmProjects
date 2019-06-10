import cv2
import numpy as np
cv2.namedWindow('image')
img = cv2.imread('1.jpg')
def thresh_demo(arg):
    y = cv2.getTrackbarPos("tarck",'image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,y,255,cv2.THRESH_BINARY)
    cv2.imshow('image',binary)
thresh = 10
cv2.createTrackbar("tarck",'image',0,255,thresh_demo)
print("#########")
cv2.waitKey()