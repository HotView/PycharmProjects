import cv2
import numpy as np
import glob
# Load previously saved data

def draw(img,corners,imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img,corner,tuple(imgpts[0].ravel()),(255,0,0),5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0,255), 5)
    return img
def drawRec(img,corners,imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    #draw ground floor in green
    img = cv2.drawContours(img,[imgpts[:4]],-1,(0,255,0),-3)
    #draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img,[imgpts[4:]],-1,(0,0,255),3)

    return img

drawflags = not True
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
if drawflags == True:
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
else:
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
with np.load('camera.npz') as X:
    mtx, dist= [X[i] for i in ('mtx','dist')]
print(mtx)
print("---")
print(dist)
for fname in glob.glob('image/left*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)
    cv2.imshow("gray",gray)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # find the rotation and translation vectors.
        flasg,rvecs,tvecs,inliers = cv2.solvePnPRansac(objp,corners2,mtx,dist)
        imgpts,jac = cv2.projectPoints(axis,rvecs,tvecs,mtx,dist)
        print(imgpts.shape)
        if drawflags == True:
            img = drawRec(img,corners2,imgpts)
        else:
            img = draw(img,corners2,imgpts)
        cv2.imshow("img",img)
        k = cv2.waitKey(0) &0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png',img)
cv2.destroyAllWindows()

