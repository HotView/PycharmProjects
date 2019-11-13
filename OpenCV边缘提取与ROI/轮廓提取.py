import cv2
import numpy as np
img = cv2.imread("image/realimage.bmp")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,177,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
cv2.imshow("thresh",thresh)
image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("con_image",contours)
count = len(contours)
print("there are ",count," contours")
for i in range(count):
    img = cv2.drawContours(img,contours,i,[0,255,0],0)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()