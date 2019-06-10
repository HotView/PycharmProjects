import cv2
import numpy as np
def erode_demo(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dst = cv2.erode(binary,kernel)
    cv2.imshow("erode",dst)

image = cv2.imread("1.jpg")
erode_demo(image)
cv2.waitKey()