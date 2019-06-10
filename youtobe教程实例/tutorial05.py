import numpy as np
import cv2
def trace_object_demo():
    capture = cv2.VideoCapture("4.mp4")
    while(True):
        ret,frame = capture.read()
        if ret == False:
            break
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_hsv =np.array([37,43,46])
        high_hsv = np.array([77, 255, 255])
        mask = cv2.inRange(hsv,lowerb=lower_hsv,upperb=high_hsv)
        cv2.imshow("mask",mask)

cv2.split()
cv2.merge()