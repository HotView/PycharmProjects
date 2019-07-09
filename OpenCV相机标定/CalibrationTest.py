import cv2
import numpy as np
import glob
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
# prepare objects points,like(0,0,0),(1,0,0),(2,0,0),....,(6,5,0)
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and points from all the images.
objpoints = []
imgpoints = []

images = glob.glob('image/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Find the chess board coners，按照行或者列连接，（7,6）就是7个连成一条线
    ret,corners = cv2.findChessboardCorners(gray,(7,6),None)
    #如果找出了角点，添加对象点和图像点
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        print("corners2",corners2)
        imgpoints.append(corners2)
        #绘制和展示角点,按照颜色来进行划分（7,6），6种颜色
        img = cv2.drawChessboardCorners(img,(7,6),corners2,ret)
        cv2.imshow(fname,img)
rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
h,w = gray.shape[:2]
imgsize = (w,h)
mtx2, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,imgsize,alpha=0,
                                     centerPrincipalPoint=True)
print("#######")
print(dist)
#np.savez("pose",mtx = mtx,dist = dist,newmtx = mtx2)
print(mtx,dist,mtx2)
with open('pose.py', 'wb') as fout:
    fout.write(b'import numpy as np\n')
    fout.write(b'\n')
    fout.write(b'rmse = %r\n' % rmse)
    fout.write(b'distortion_coefficients = np.array(%r, dtype=np.float32)\n'
               % dist.tolist())
    fout.write(b'raw_camera_matrix = np.array(%r, dtype=np.float32)\n'
               % mtx.tolist())
    fout.write(b'undistort_camera_matrix = np.array(%r, dtype=np.float32)\n'
               % mtx2.tolist())
    fout.write(b'roi = %d, %d, %d, %d\n'% roi)
    fout.write(b'image_size = %d, %d\n' % imgsize)
print(roi)
print("----------------")

print(ret)
print("-----------")
print(mtx)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print ("total error: ", mean_error/len(objpoints))
cv2.waitKey(0)




