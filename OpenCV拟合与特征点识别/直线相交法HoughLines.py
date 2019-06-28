import cv2
import numpy as np
def ResLine(lines):
    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b= np.sin(theta)
        x0 = a*rho
        y0 = b*rho

img = cv2.imread("test01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
lines = cv2.HoughLines(gray,1.0,np.pi/180,150)
for line in lines:
    print(line)