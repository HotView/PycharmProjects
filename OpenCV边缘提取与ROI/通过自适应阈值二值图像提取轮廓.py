import cv2
import numpy as np
from OpenCV中心线提取.ExtCenterPts import Extreme,ExtremeBound
from OpenCV相机标定.FitLine import getFitline
from OpenCV边缘提取与ROI.GetMaxRegion import getmaxRectThresh
from scipy.optimize import leastsq


img = cv2.imread("chessimg/laser01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x,y,w,h = getmaxRectThresh(gray)
#---------------------------------
mask = np.zeros(img.shape[:2],np.uint8)
mask[y:(y+h),x:(x+w)]=255
#----------------------
ROI = cv2.bitwise_and(gray,gray,mask=mask)
points = np.array(ExtremeBound(ROI,x+10,x+w-10))
k,b = getFitline(points)
point1 = (0,int(b))
point2 = (2000,int(2000*k+b))
#------------------------
cv2.line(img,point1,point2,[0,255,0],1)
cv2.imshow("origin",img)
cv2.waitKey(0)

