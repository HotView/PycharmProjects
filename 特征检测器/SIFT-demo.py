import cv2
import sys
import numpy as np

img = cv2.imread('demo02.jpg')
img = cv2.resize(img,(600,800))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints,descriptor = sift.detectAndCompute(gray,None)
print(len(keypoints))
print(descriptor)
print(descriptor.shape)
img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(51,163,236))

cv2.imshow('sift_keypoint',img)
while (True):
    if cv2.waitKey(10)&0xff == ord("q"):
        break
cv2.destroyAllWindows()
