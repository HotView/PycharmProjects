import numpy as np
import cv2
img = cv2.imread("test_rec.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow("test",0)
cv2.namedWindow("detector",0)
Ix = cv2.Scharr(gray,cv2.CV_32F,1,0)
Iy = cv2.Scharr(gray,cv2.CV_32F,0,1)
Ixx = cv2.Scharr(Ix,cv2.CV_32F,1,0)
Ixy = cv2.Scharr(Ix,cv2.CV_32F,0,1)
Iyy = cv2.Scharr(Iy,cv2.CV_32F,0,1)
Iyx = cv2.Scharr(Iy,cv2.CV_32F,1,0)
detector = (cv2.GaussianBlur(Ixy,(5,5),0)*cv2.GaussianBlur(Iyx,(5,5),0)- cv2.GaussianBlur(Ixx,(5,5),0)*cv2.GaussianBlur(Iyy,(5,5),0))
points = cv2.goodFeaturesToTrack(detector,maxCorners=64,qualityLevel=0.1,minDistance=5,blockSize=3)
print(points.shape)
print(np.max(detector,axis=0))
points_int = points.astype(np.int).reshape(36,2)
print(points_int)
x_index = points_int[:,0]
print(x_index)
y_index = points_int[:,1]
dhfjhs = img[x_index,y_index,:]
print(dhfjhs)
img[x_index,y_index,:]= [0,255,0]
cv2.imshow("test",img)
cv2.imshow("detector",detector)
cv2.waitKey()