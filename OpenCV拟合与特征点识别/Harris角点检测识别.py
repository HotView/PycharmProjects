import cv2
import numpy as np

img = cv2.imread("center-v.jpg")
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dist = cv2.cornerHarris(gray,2,3,0.04)
dist = cv2.dilate(dist,None)
distmax = dist.max()
img[dist>0.01*distmax] = [0,0,255]
cv2.imshow("origin",img)
cv2.waitKey(0)