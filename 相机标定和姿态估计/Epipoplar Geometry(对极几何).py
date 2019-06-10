import cv2
import numpy as np
from matplotlib import pyplot as plt
def drawlines(img1,img2,lines,pts1,pts2,good):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int,[0,-r[2]/r[1]])
        x1,y1  =map(int,[c,-(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1,(x0,y0),(x1,y1),color,1)
        img1  = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt1),5,color,-1)

    return img1,img2


img1 = cv2.imread("image/1_01.jpg",0)
img2 = cv2.imread("image/1_02.jpg",0)

sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k =2)
matches = sorted(matches,key=lambda x:x[0].distance)
matches = matches[0:100]
good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
    if m.distance<0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
## 只选择孤立点
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
lines1 = lines1.reshape(-1,3)
img3,img4,img5 = drawlines(img1,img2,lines1,pts1,pts2,good)

cv2.namedWindow("img3",0)
cv2.namedWindow("img4",0)
cv2.namedWindow("img5",0)
cv2.imshow("img3",img3)
cv2.imshow("img4",img4)
cv2.imshow("img5",img5)
cv2.waitKey(0)
cv2.destroyAllWindows()


