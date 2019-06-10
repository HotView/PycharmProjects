import cv2
import numpy as np
R = np.array([[ 1.,   0.,   0.  ] ,[ 0.,   0.99609879, -0.08824514], [-0.,   0.08824514, 0.99609879]] )
Rodrigues = cv2.Rodrigues(R,jacobian=0)
print(R)
print(0.08836007/3.14*180)
for one in Rodrigues:
    print("--------------")
    print(one)
Vec = np.array([[0.08836007,0,0]])
Rodrigues1 = cv2.Rodrigues(Vec)
for one in Rodrigues1:
    print("--------------")
    print(one)
