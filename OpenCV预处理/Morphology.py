import cv2
import numpy as np
def Resize(img):
    res = cv2.resize(img,(600,800))
    cv2.imwrite("laser02.jpg",res)

def Erode(gray_img,ksize = 5):
    kernel = np.ones((ksize,ksize),np.uint8)
    erosion = cv2.erode(gray_img,kernel,iterations=1)
    return erosion
def Dilation(gray_img,ksize = 5):
    kernel = np.ones((ksize, ksize), np.uint8)
    dilation = cv2.dilate(gray_img, kernel, iterations=1)
    return dilation
def Opening(gray_img,ksize = 5):
    """
    去除外部的零散点
    :param gray_img:
    :param ksize:
    :return:
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    return opening
def Closig(gray_img,ksize = 5):
    """
    #填充前景物中的小洞，或者小黑点
    :param gray_img:
    :param ksize:
    :return:
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    closing = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
    return closing
