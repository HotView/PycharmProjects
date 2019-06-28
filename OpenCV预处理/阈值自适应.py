import cv2
import numpy as np
def Nothing(val):
    size = cv2.getTrackbarPos("size","gray")
    param = cv2.getTrackbarPos("param","gray")
    thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,2*size+1, -param)
    cv2.imshow("thresh", thresh)
img = cv2.imread("hand01.jpg")
img = cv2.GaussianBlur(img,(5,5),1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("thresh")
cv2.namedWindow("gray")
cv2.createTrackbar("size","gray",0,300,Nothing)
cv2.createTrackbar("param","gray",0,100,Nothing)
cv2.imshow("gray",gray)
cv2.waitKey(0)