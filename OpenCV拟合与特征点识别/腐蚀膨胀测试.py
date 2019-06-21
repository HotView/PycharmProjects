import cv2
import numpy as np

img = cv2.imread("center-v.jpg")
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("origin",gray)
#dilation = cv2.dilate(gray,(3,3))
#erode = cv2.erode(gray,(3,3))
#res = dilation+erode
#cv2.imshow("dialtion",dilation)
#cv2.imshow("erode",erode)

#cv2.imshow("res",res)
cv2.waitKey(0)
