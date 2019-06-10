import cv2
import numpy as np

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
objp = np.zeros((4*6,3),np.float32)
objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)

objpoints = []# 3d point in real world space.
imgpoints = []# 2d points in image plane.

img = cv2.imread('Calibration.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(img.shape)
ret,corners  = cv2.findChessboardCorners(img_gray,(6,4),None)
print(ret)
print(corners)
if ret ==True:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(img_gray,corners,(11,11),(-1,-1),criteria=criteria)
    imgpoints.append(corners2)

    img_cali = cv2.drawChessboardCorners(img,(6,4),corners2,ret)
cv2.imshow("origin",img)
cv2.imshow("calibration",img_cali)
cv2.waitKey()
cv2.destroyAllWindows()