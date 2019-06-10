#管理滤波器的类
import cv2
import numpy as np
a= np.array([-511,525,45,41,1,-32])
unt8 = cv2.convertScaleAbs(a)
print(unt8)


