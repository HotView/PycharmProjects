import numpy as np
import cv2
import math
def threshEntroy(image):
    rows,cols = image.shape
    grayHist = cv2.calcHist(image)
    normGrayHist = grayHist/float(rows*cols)
    zeroCumuMoment = np.zeros([256],np.float32)
    #计算累加直方图
    for k in range(256):
        if k==0:
            zeroCumuMoment[k] = normGrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k-1]+normGrayHist[k]
    # hui du ji entroy
    entroy = np.zeros([256],np.float32)
    for k in range(256):
        if k==0:
            if normGrayHist[k] == 0:
                entroy[k] = 0
            else:
                entroy[k] = -normGrayHist[k]*math.log10(normGrayHist[k])
        else:
            if normGrayHist[k]==0:
                entroy[k] = entroy[k-1]
            else:
                entroy[k] = entroy[k-1]-normGrayHist[k]*math.log10(normGrayHist[k])
    fT = np.zeros([256],np.float32)

x = np.arange(9).reshape(3, 3)
y = np.bitwise_and(x,1)
print(x)
print(y)