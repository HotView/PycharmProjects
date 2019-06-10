#即把要拼接的图片经过透视变换，得到其在基准图像中的位置，然后用基准图像覆盖即可。
#记得M = findHomography函数的输出，即输入点右乘M得到输出点。
## 使用FLANN匹配器注意描述符的匹配问题
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

GOOG_POINT_LIMITED = 0.99
src = 'Images/1_01.jpg'
des = 'Images/1_02.jpg'
img1_01 = cv2.imread(src,1)#基准图像
img1_gray = cv2.cvtColor(img1_01,cv2.COLOR_BGR2GRAY)
img1_02 = cv2.imread(des,1)#拼接图像
img2_gray  =cv2.cvtColor(img1_02,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1_01,None)
kp2,des2 = orb.detectAndCompute(img1_02,None)
if des1.dtype is not cv2.CV_32F:
    des1.astype(np.float32)
if des2.dtype is not cv2.CV_32F:
    des2.astype(np.float32)
#FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1)
search_params = dict(checks=50) # or pass empty dictionary

fbm = cv2.FlannBasedMatcher(index_params,search_params)
matches = fbm.knnMatch(des1,des2,k=2)
#matchMask = [[0,0] for i in range(len(matches))]
## 调试寻找最优的匹配点。
goodPoints = []
for i,(m,n) in enumerate(matches):
    if m.distance<0.7*n.distance:
        goodPoints.append(m)
        #matchMask[i] = [1,0]
print("最优的匹配点的个数：",len(goodPoints))
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),flags = 0)
img3 = cv2.drawMatches(img1_01,kp1,img1_02,kp2,goodPoints,None,**draw_params)


MIN_MATCH_COUNT = 10
if len(goodPoints)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1,1,2)

    M,mask = cv2.findHomography(dst_pts,src_pts,cv2.RANSAC,5.0)

    h,w = img1_gray.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst  =cv2.perspectiveTransform(pts,M)
    img2_gray = cv2.polylines(img1_gray,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    plt.figure()
    plt.imshow(img2_gray,'gray')
    #cv2.imshow("original_image_overlapping.jpg",img1_02)
else:
    print("Not enough matches are found - %d/%d" % (len(goodPoints), MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img1_02,M,(img1_01.shape[1]+img1_02.shape[1],img1_01.shape[0]))
plt.figure()
plt.imshow(dst)
dst[0:img1_01.shape[0],0:img1_01.shape[1]] = img1_01
plt.figure()
plt.imshow(dst)

plt.figure()
plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
plt.show()