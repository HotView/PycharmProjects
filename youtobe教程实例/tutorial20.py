import cv2
import numpy as np
def edge_demo(image):
    blurred = cv2.GaussianBlur(image,(3,3),0)
    gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
    xgrad = cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0,1)
    cv2.CV_32
    edg_out = cv2.Canny(xgrad,ygrad,50,150)
    cv2.imshow("Canny image",edg_out)
    dst = cv2.bitwise_and(image,image,mask=edg_out)
    cv2.imshow("Canny color image",dst)
    cv2.imshow("image",image)
image = cv2.imread("4.jpg")
edge_demo(image)
cv2.waitKey()