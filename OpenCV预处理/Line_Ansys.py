import cv2
import numpy as np

img = cv2.imread("laser_test00.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
newimg= img[1800:2100,:,:].copy()
cv2.namedWindow("gray",0)
print(newimg.shape)
cv2.imshow("gray",newimg)
np.savetxt("laser_test.txt",gray[1800:2100,500:900],fmt="%.3d")
row,col,channel = img.shape
print(row,col)
for j in range(col):
    print(np.argmax(gray[:,j]),j)
cv2.waitKey(0)