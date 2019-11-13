import cv2
import numpy as np
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
# prepare objects points,like(0,0,0),(1,0,0),(2,0,0),....,(6,5,0)
num_line = 6
num_point = 9
objp = np.zeros((num_line*num_point,3),np.float32)
objp[:,:2] = np.mgrid[0:num_point,0:num_line].T.reshape(-1,2)
print("###################################################")
print(objp)
print("###################################################")
# Arrays to store object points and points from all the images.
objpoints = []
imgpoints = []


images = glob.glob('../camera/chess/chess*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    #Find the chess board coners，按照行或者列连接，（7,6）就是7个连成一条线
    ret,corners = cv2.findChessboardCorners(gray,(num_point,num_line),None)
    #如果找出了角点，添加对象点和图像点
    if ret:
        objpoints.append(objp)
        print("ret is true")
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #print("corners2",corners2)
        imgpoints.append(corners2)
        #绘制和展示角点,按照颜色来进行划分（7,6），6种颜色
        img = cv2.drawChessboardCorners(img,(num_point,num_line),corners2,ret)
        cv2.imshow(fname,img)
rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
h,w = gray.shape[:2]
imgsize = (w,h)
mtx2, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,imgsize,alpha=0,
                                     centerPrincipalPoint=True)
print("#######")
print("dist= ",dist)
np.savez("../camera/camera",mtx = mtx,dist = dist,newmtx = mtx2)
print("mtx =",mtx,"dist=",dist,"mtx2= ",mtx2)
print("----------------")
print("-----------")
print("mtx",mtx)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print ("total error: ", mean_error/len(objpoints))
cv2.waitKey(0)




