import numpy as np
import cv2
img = cv2.imread("test03.jpg")
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
print(type(detector))
slice1 = np.array([1,5,8,6,2])
slice2 = np.array([1,5,8,6,2])
print(img[slice1,slice2])
print(np.max(detector),np.min(detector))

detector_norm = np.zeros(detector.shape)
cv2.normalize(np.abs(detector),detector_norm,0,255,cv2.NORM_MINMAX)
points = cv2.goodFeaturesToTrack(detector,maxCorners=100,qualityLevel=0.1,minDistance=5,blockSize=3)
points_int = points.astype(np.int).reshape((-1,2))
print(np.where(detector_norm))
cv2.namedWindow("normimage",0)
cv2.imshow("normimage",detector_norm)
for center in points_int:
    i,j=center
    img[j,i] = [255,0,0]
    V = np.linalg.eig(np.array([[Ixx[j,i],Ixy[j,i]],[Iyx[j,i],Iyy[j,i]]]))[1]
    V_change = np.dot([[1,1],[1,-1]],V.T)
    V_change = (V_change*10).astype(np.int)
    V= V*10
    V = V.astype(np.int)
    cv2.arrowedLine(img,(center[0],center[1]),(center[0]+V[0,0],center[1]+V[0,1]),(0,0,255),1,cv2.LINE_8)
    cv2.arrowedLine(img,(center[0],center[1]),(center[0]+V[1,0],center[1]+V[1,1]),(0,255,0),1)
    cv2.arrowedLine(img,(center[0],center[1]),(center[0]+V_change[0,0],center[1]+V_change[0,1]),(0,255,255),1)
    cv2.arrowedLine(img,(center[0],center[1]),(center[0]+V_change[1,0],center[1]+V_change[1,1]),(255,0,0),1)
cv2.arrowedLine(img,(51,21),(154,186),(255,0,0),1,cv2.LINE_AA)
cv2.imshow("test",img)
cv2.imshow("detector",detector)
cv2.waitKey()