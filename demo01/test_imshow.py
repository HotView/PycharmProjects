import cv2
import numpy as np
img = cv2.imread('3.jpg')
while True:
    cv2.imshow("img",img)
    cv2.waitKey(2000)
    cv2.resizeWindow("img",100,100)
np.zeros()
