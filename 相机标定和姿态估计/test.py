import numpy as np
import cv2

img = cv2.imread("meizu01.jpg",0)
img[200:600,:] = -700
img[600:800,:] = -255
img[800:1200,:] = -0
cv2.namedWindow('winname',0)
cv2.imshow('winname',img)
cv2.waitKey()
cv2.transform()
