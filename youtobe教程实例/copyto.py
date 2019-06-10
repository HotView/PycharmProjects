import cv2
import numpy as np
cv2.namedWindow('image')
img = cv2.imread('1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
test = img.copyto()
print("#########")
cv2.waitKey()