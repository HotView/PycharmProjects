import numpy as np
import cv2

img = cv2.imread("laser_bin.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
arg_row = np.argwhere(gray>0)
Min = np.min(arg_row,axis=0)
Max = np.max(arg_row,axis=0)
pt1 = (Min[1],Min[0])
pt2 = (Max[1],Max[0])
hahah = img[Min[0]:Max[0],Min[1]:Max[1],:]
cv2.imshow("haha",hahah)
hahah[0:10,0:10,:] = 40
cv2.rectangle(img,pt1,pt2,(0,255,0))
cv2.imshow("img",img)

print(Min,Max)
print(arg_row.shape)
print(arg_row)
cv2.waitKey(0)