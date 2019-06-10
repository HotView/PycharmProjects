import cv2
import numpy as np
import glob

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

# prepare object points,like (0,0,0),(1,0,0),(2,0,0)....(6,5,0)
#objp = np.mgrid[0:7,0:6]
#objp = objp.reshape(-1,2)
#print(objp)
objp = np.zeros((6*7,3),np.float32)
print(np.mgrid[0:7,0:6].T.reshape(-1,2))
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
print(objp)