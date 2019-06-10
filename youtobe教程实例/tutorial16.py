#threshold操作
#阈值操作
import cv2
import numpy as np
def threshold(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    print("threshold:",ret)
    cv2.imshow("binary",binary)
def local_threshold(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,123,10)
    cv2.imshow("local-binary", binary)
image = cv2.imread("1.jpg")
cv2.imshow("origin image",image)
local_threshold(image)
threshold(image)
cv2.waitKey()
cv2.calcHist(1,)
