import cv2
import numpy as np
from OpenCV相机标定.Get3dPoints import get3dPoints
import glob

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
with np.load('../camera/laserparams.npz') as X:
    lineparams =X["params"]
num_chessline = 6
num_linepoint = 9
objp = np.zeros((num_chessline*num_linepoint,3),np.float32)
objp[:,:2] = np.mgrid[0:num_linepoint,0:num_chessline].T.reshape(-1,2)
objp = objp.reshape((num_chessline,num_linepoint,3))
index = 0
fname =glob.glob("../camera/chess/ch*.jpg")[index]
print(fname)
laser_line = lineparams[index]
img = cv2.imread(fname)
#cv2.imshow("origin",img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners  =cv2.findChessboardCorners(gray,(num_linepoint,num_chessline),None)
if ret:
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    img = cv2.drawChessboardCorners(img, (num_linepoint, num_chessline), corners2, ret)
    corners3= corners2.reshape((num_chessline,num_linepoint,2))
    k,b = laser_line
    point1 = (0,int(b))
    point2 = (2000,int(2000*k+b))
    cv2.line(img,point1,point2,[0,255,0],1)
    points3d,points_neigh,cross_points,close_points = get3dPoints(laser_line,corners3,objp)
    for points_lines in points_neigh:
        for one in points_lines:
            #print(tuple(one))
            cv2.circle(img,tuple(one),3,[0,0,255],-1)
    print(np.array(points3d))
    print(np.array(points_neigh).shape)
    print(np.array(cross_points).shape)
    for index in range(len(cross_points)):
        cv2.line(img,tuple(cross_points[index].astype(np.int)),tuple(close_points[index].astype(np.int)),[250,255,0],2)
    cv2.imshow(fname, img)
cv2.waitKey(0)
