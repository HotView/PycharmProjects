##直方图均衡化
###对象是灰度图像
import cv2
import  numpy as np
def equalHist(images):
    gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    cv2.imshow("zhifangtu",dst)

def adaptive_equalHist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe= cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    dst = clahe.apply(gray)
    cv2.imshow("自适应直方图",dst)

def creat_RGB_hist(image):
    h,w,c = image.shape
    rgbHist = np.zeros([16*16*16,1],np.float32)
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row,col,0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16+np.int(g/bsize)*16+np.int(r/bsize)
            rgbHist[np.int(index),0] = rgbHist[np.int(index),0]+1
    return rgbHist
imsge = cv2.imread("1.jpg")
equalHist(imsge)
adaptive_equalHist(imsge)
cv2.waitKey()