#即把要拼接的图片经过透视变换，得到其在基准图像中的位置，然后用基准图像覆盖即可。
#记得M = findHomography函数的输出，即输入点右乘M得到输出点。
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

GOOG_POINT_LIMITED = 0.99
src = 'Images/1_01.jpg'
des = 'Images/1_02.jpg'
img1_01 = cv2.imread(src,1)#基准图像
img1_02 = cv2.imread(des,1)#拼接图像

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1_01,None)
kp2,des2 = orb.detectAndCompute(img1_02,None)

bf = cv2.BFMatcher.create()
matches = bf.knnMatch(des1,des2,k=2)
#matches = sorted(matches,key=lambda x:x.distance)。
## 调试寻找最优的匹配点。
goodPoints =[]
for m,n in matches:
    if m.distance<0.6*n.distance:
        goodPoints.append(m)
for one in goodPoints:
    print(one.distance)
print("最优的匹配点的个数：",len(goodPoints))
draw_Params = dict(matchColor = (0,255,0),singlePointColor =  None,flags = 2)
img3 = cv2.drawMatches(img1_01,kp1,img1_02,kp2,goodPoints,None,**draw_Params)
plt.figure()
plt.imshow(cv2.cvtColor(img3,cv2.COLOR_BGR2RGB))
plt.show()
'''
#图像配准，findHomography函数所要用到的点集是Pointf类型的，
#所以需要对我们得到的点集再做一次处理，使其转换为Pointf类型的点集
src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1,1,2)
# 这里选择RHO算法继续筛选可靠的匹配点，使得匹配点更精准。
M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RHO)
print(M)
print(mask)
# 获取原图像的高和宽
h1,w1,p1 = img1_01.shape
h2,w2,p2 = img1_02.shape
h = np.maximum(h1,h2)
w = np.maximum(w1,w2)

_movedis = int(np.maximum(dst_pts[0][0][0],src_pts[0][0][0]))
imageTransform = cv2.warpPerspective(img1_02,M,(w1+w2-_movedis,h))

M1 = np.float32([[1,0,0],[0,1,0]])
h_1,w_1,p = img1_01.shape
dst1 = cv2.warpAffine(img1_01,M1,(w1+w2-_movedis,h))


dst = cv2.add(dst1,imageTransform)
dst_no = np.copy(dst)

dst_target = np.maximum(dst1,imageTransform)
fig = plt.figure(tight_layout = True,figsize = (8,18))
gs = gridspec.GridSpec(6,2)
ax = fig.add_subplot (gs[0, 0])
ax.imshow(img1_01)
ax = fig.add_subplot (gs[0, 1])
ax.imshow(img1_02)
ax = fig.add_subplot (gs[1, :])
ax.imshow(img3)
ax = fig.add_subplot (gs[2, :])
ax.imshow(imageTransform)
ax = fig.add_subplot (gs[3, :])
ax.imshow(dst1)
ax = fig.add_subplot (gs[4, :])
ax.imshow(dst_no)
ax = fig.add_subplot (gs[5, :])
ax.imshow(dst_target)
ax.set_xlabel ('The smooth method is SO FAST !!!!')
plt.figure(2)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
plt.imshow(img3)

plt.show()
'''