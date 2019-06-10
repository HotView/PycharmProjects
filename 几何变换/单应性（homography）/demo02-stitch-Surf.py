import cv2
import numpy as np
from matplotlib import pyplot as plt
img_top = cv2.imread('Images/demo01.jpg')
print('img_right size :',img_top.shape)
img_top_gray = cv2.cvtColor(img_top,cv2.COLOR_BGR2GRAY)

img_bot = cv2.imread('Images/demo02.jpg')
img_bot_gray = cv2.cvtColor(img_bot,cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
# find key points
kp1, des1 = surf.detectAndCompute(img_top_gray,None)
kp2, des2 = surf.detectAndCompute(img_bot_gray,None)

#cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))

#FLANN_INDEX_KDTREE = 0
#index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#search_params = dict(checks = 50)
#match = cv2.FlannBasedMatcher(index_params, search_params)
match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.18*n.distance:
        good.append(m)
print("特征点匹配的对数：",len(good))

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(img_top,kp1,img_top,kp2,good,None,**draw_params)
#cv2.imshow("original_image_drawMatches.jpg", img3)
#good = good[0:4]
MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
    print(len(src_pts))
    M,mask = cv2.findHomography(dst_pts, src_pts,cv2.RANSAC,5.0)
    print("findHo-M:",M)
    print("输入序列点的长度：",len(src_pts))
    print("len(mask) :",len(mask))
    print("mask", mask)
    h,w = img_top_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    print(pts)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img_top_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
    cv2.namedWindow('windows',0)
    cv2.resizeWindow('windows', (800, 600))
    cv2.imshow('windows', img2)

else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img_bot ,M,(img_top.shape[1]+1000, img_top.shape[0]+1000))
plt.figure(2)
dst2 = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
plt.imshow(dst2)
#cv2.imshow('warpPerspective', dst)
dst[0:2000,0:img_top.shape[1]] = img_top[0:2000,:]
plt.figure(3)
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)
plt.imshow(dst)
plt.show()
