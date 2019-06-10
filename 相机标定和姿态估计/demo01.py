import cv2
import numpy as np
import glob

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

# prepare object points,like (0,0,0),(1,0,0),(2,0,0)....(6,5,0)
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)


objpoints = []#3d point in real world space
imgpoints = []#2d points in image plane

images = glob.glob('data\left*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    print(gray.shape[::-1])
    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)
    #print(corners)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #print(corners2-corners)
        imgpoints.append(corners2)

        #Draw and display the corners

        img = cv2.drawChessboardCorners(img,(7,6),corners2,ret)
        cv2.imshow(fname,img)
        cv2.waitKey(500)

a = cv2.waitKey()
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
