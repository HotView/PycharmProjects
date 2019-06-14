import cv2
import numpy as np
import sys
import os
img = cv2.imread('3.jpg')
print(sys.executable,"###")
script_path = os.path.abspath(__file__)
workdir = os.path.abspath('.')
print(script_path)
print(workdir)
"""
while True:
    cv2.imshow("img",img)
    cv2.waitKey(2000)
    cv2.resizeWindow("img",100,100)
np.zeros()
print(sys.executable)
"""
