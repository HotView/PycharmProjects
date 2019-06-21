import cv2
import numpy as np
def test():
    a = np.ones((10,10))
    b = a[5:8,5:8]
    c = a[5:8,5:8].copy()
    b[0:3,0:3] = 10
    c[0:3,0:3] = 6
    print(c)
    print(a)
def test_if(num):
    a=5 if num>0 else 6
    print(a)
b= 0
test_if(5)