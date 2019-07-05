import cv2
import numpy as np
import glob
import os.path


with np.load('camera.npz') as X:
    mtx, dist= [X[i] for i in ('mtx','dist')]
img = cv2.imread("image/left12.jpg")
h,w = img.shape[:2]
newcamera_mtx,roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
images = glob.glob('image/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    cv2.imshow(fname+"1",img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst = cv2.undistort(img, mtx, dist, None, newcamera_mtx)# crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    name = os.path.split(fname)
    dstname = os.path.join('Result',name[1])
    print(dstname)
    #testname = 'Result/left01.jpg'
    cv2.imwrite(dstname, dst)
    #cv2.imshow(fname,dst)
#cv2.waitKey()
