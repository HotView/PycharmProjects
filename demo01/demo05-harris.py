import cv2
import numpy as np

img = cv2.imread("book01resize.jpg")
img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
float =np.float32(img_gray)
print(type(float[0,0]))
harris = cv2.cornerHarris(float,2,3,0.04)
harris = cv2.dilate(harris,None)
#cv2.imshow("dhfjosi",img)
j = 0
print(type(harris))
print(harris)
print(harris.shape)
'''for i in harris:
    if i > 0.1*harris.max():
        j = j+1
print(j)'''
img[harris>0.1*harris.max()] = [0,255,0]
cv2.imshow("harris",img)

cv2.waitKey()
cv2.destroyAllWindows()
