# 测试像素点的显示
import numpy as np
import cv2
cv2.namedWindow("img",0)
img = np.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,-3,-3,-3,-3,-3,254,254,254,254,254] for i in range(10)],dtype=np.int8)
print(img)
cv2.imshow("img",img)
cv2.waitKey(1000)
print("f-----------")
a= np.linspace(-256,256,256*2+1)
b = np.array(a,dtype=np.int8)
print(b)
np.array()