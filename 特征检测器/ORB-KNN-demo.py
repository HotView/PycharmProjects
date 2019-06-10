import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('mat01.jpg',0)
img2 = cv2.imread('mat02.jpg',0)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = False)
matches = bf.knnMatch(des1,des2,k=2)

<<<<<<< HEAD
print(matches[0])
=======
>>>>>>> 61326c24919029217a1e3913187e4d81184d980e
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,img2,flags=2)
plt.imshow(img3)
plt.show()