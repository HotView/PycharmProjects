import glob
import cv2
import matplotlib.pyplot as plt

cv2.findChessboardCorners()
filenames = glob.glob('data-meizu/*.jpg')
print(len(filenames))
dir_test = filenames[1]
img = cv2.imread(dir_test)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(img_gray, maxCorners=59 * 59, qualityLevel=0.1, minDistance=5, blockSize=3)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

plt.imshow(img)
plt.show()
"""cv2.namedWindow('img',0)
cv2.imshow('img',img)
cv2.waitKey()
"""
