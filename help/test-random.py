import numpy as np
import cv2
cv2.namedWindow("waitkey")
while True:
    key = cv2.waitKey(1000)
    print(type(key))
    print(key)
    print(key&0xff)

#a = np.random.randn(5,5)
#print(a)q