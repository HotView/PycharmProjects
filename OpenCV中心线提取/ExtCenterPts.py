import cv2
import numpy as np
def ExtCenter(gray):
    row,col = gray.shape
    #gray = cv2.cvtColor(,cv2.COLOR_BGR2GRAY)
    points= []
    #newimage = np.zeros((row,col),np.uint8)
    for i in range(col):
        Prow = np.argmax(gray[:,i])
        points.append([Prow,i])
    return points
def ExtremeCenter(gray,left= 0,right=0) :
    """
    return center points
    :param gray:input image
    :param left: left bounding
    :param right: right bounding
    :return: center point
    """
    row, col = gray.shape
    if left==0 and right==0:
        right = col
    #gray = cv2.cvtColor(,cv2.COLOR_BGR2GRAY)
    points= []
    print(col)
    #newimage = np.zeros((row,col),np.uint8)
    for i in range(col):
        if i>=left and i<=right:
            Prow = np.argmax(gray[:,i])
            points.append([i,Prow])
    return points