import cv2
import numpy as np
a= np.array([[i for i in range(3)]for j in range(3)])
b = np.array([[i for i in range(4)]for j in range(3)])
b= np.array([[1,2,3],[4,5,6],[7,8,9]])
c = a*b
print(a)
print(c)