import cv2
import numpy as np
def Resize(img,filename):
    row,col,channel = img.shape
    c_row = int(row/2)
    c_col = int(col/2)
    print(c_row,c_col)
    res = cv2.resize(img,(c_col,c_row))
    cv2.imwrite("E:/PaperImage/"+filename,res)
filename = ""
img = cv2.imread(filename)
Resize(img,filename)