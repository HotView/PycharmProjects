import cv2
import numpy as np

img = cv2.imread("hand01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)