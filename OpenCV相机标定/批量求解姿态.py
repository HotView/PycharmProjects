import cv2
import numpy as np
import glob

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
num_chessline = 6
num_linepoint = 9
objp = np.zeros((num_chessline*num_linepoint,3), np.float32)
objp[:,:2] = np.mgrid[0:num_linepoint,0:num_chessline].T.reshape(-1,2)
with np.load('../camera/camera.npz') as X:
    newmtx, dist= [X[i] for i in ('newmtx','dist')]
pose = []
for fname in glob.glob('../camera/chess/chess*.jpg'):
#fname = "../camera/chess/chess01.jpg"
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,corners = cv2.findChessboardCorners(gray,(num_linepoint,num_chessline),None)
    cv2.imshow("gray",gray)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # find the rotation and translation vectors.
        flags,rvecs,tvecs,inliers = cv2.solvePnPRansac(objp,corners2,newmtx,dist)
        #print("rvecs",rvecs)
        rotate3d = cv2.Rodrigues(rvecs)[0]
        #print("fjdks",rotate3d)
        #print(flags)
        #print(tvecs)
        #print("----")
        TransMat = np.hstack([rotate3d,tvecs])
        pose.append(TransMat)
        """
        points1 = np.array([1,0,0,1]).reshape(4,1)
        res = np.dot(TransMat,points1)
        print("###############")
        print(res)
        #print(inliers)
        print("###############")
        """
print(pose)
np.savez("../camera/poses",poses = pose)
#data = np.load("../camera/camera.npz")
#print(data["poses"])
#for x in data.keys():
#    print(x)