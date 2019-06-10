import glob
import cv2
import matplotlib.pyplot as plt


filenames = glob.glob('data-meizu/*.jpg')
print(len(filenames))
dir_test = filenames[1]
img = cv2.imread(dir_test)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img_gray, None)
cv2.drawKeypoints(img, outImage=img, keypoints=kp)
plt.imshow(img)
plt.show()
"""cv2.namedWindow('img',0)
cv2.imshow('img',img)
cv2.waitKey()
"""
