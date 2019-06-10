import cv2
import sys
import numpy as np

img = cv2.imread('demo02.jpg')
img = cv2.resize(img,(600,800))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(8000)
keypoints,descriptor = surf.detectAndCompute(gray,None)
print(len(keypoints))

img = cv2.drawKeypoints(image=img,outImage=img,keypoints=keypoints,flags=4,color=(51,163,236))

cv2.imshow('keypoints',img)
while (True):
    if cv2.waitKey(1000) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()