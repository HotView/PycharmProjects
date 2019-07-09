import cv2
import numpy as np
from OpenCV中心线提取.ExtCenterPts import ExtremeCenter
from OpenCV相机标定.FitLine import getFitline
from OpenCV边缘提取与ROI.GetMaxRegion import getmaxRectThresh
import glob


params = []
fnames = glob.glob("../camera/chess/la*.jpg")
for i,fname in enumerate(fnames):
    #fname = os.path.abspath(fname)
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #------------------------------------
    x,y,w,h = getmaxRectThresh(gray)
    cv2.rectangle(img,(x,y),(x+w,y+h),[255,0,0],1)
    # ----------------------
    mask = np.zeros(img.shape[:2],np.uint8)
    mask[y:(y+h),x:(x+w)]=255
    ROI = gray[y:(y+h),x:(x+w)]
    points = np.array(ExtremeCenter(ROI,20,w-20))+[x,y]
    k,b = getFitline(points)
    #-------------------------
    print(k,b)
    params.append([k,b])
    #parapath = "../pose/laserpara0"+str(i+1)
    #np.savez(parapath,k = k,b = b)
    #------------------------
    point1 = (0,int(b))
    point2 = (2000,int(2000*k+b))
    cv2.line(img,point1,point2,[0,255,0],1)
    cv2.imshow(fname,img)
#cv2.imshow("ROI",ROI)
print(params)
np.savez("../camera/laserparams",params = params)
cv2.waitKey(0)


