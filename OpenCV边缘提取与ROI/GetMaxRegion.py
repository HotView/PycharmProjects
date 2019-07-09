import numpy as np
import cv2

def getmaxRectThresh(gray):
    """
    通过阈值二值化来获取轮廓区域
    :param gray:input image
    :return: ROI rect
    """
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # ---------------------
    kernel_dilate = np.ones((10,10),np.uint8)
    kernel = np.ones((20, 20), np.uint8)
    dilatetion = cv2.dilate(thresh,kernel_dilate)
    morph = cv2.morphologyEx(dilatetion, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_region = 0
    max_rect = []
    for obj in contours:
        x,y,w,h = cv2.boundingRect(obj)
        region = w*h
        if region>max_region:
            max_region = region
            max_rect= [x,y,w,h]
        #vertices = cv2.boxPoints(rotatedRect)
    return max_rect
def getmaxRectEdge(gray):
    """
    通过Canny边缘算子来获取轮廓区域
    :param gray:input image
    :return: ROI rect
    """
    threshold1 = 100
    threshold2 = 300
    kernel = np.ones((25, 25), np.uint8)
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel=kernel)
    cannyimg = cv2.Canny(morph, threshold1, threshold2)
    # ---------------------
    morph = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
    _, contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_region = 0
    max_rect = []
    for obj in contours:
        x,y,w,h = cv2.boundingRect(obj)
        region = w*h
        if region>max_region:
            max_region = region
            max_rect= [x,y,w,h]
        #vertices = cv2.boxPoints(rotatedRect)
    return max_rect