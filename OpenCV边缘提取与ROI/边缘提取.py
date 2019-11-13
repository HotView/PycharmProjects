import cv2
import numpy as np
img = cv2.imread("image/realimage.bmp")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
edge = np.sqrt(np.power(sobelx,2)+np.power(sobely,2))
edge = edge/np.max(edge)
edge = np.power(edge,1)
edge*=255
edge = edge.astype(np.uint8)

cv2.imshow("sobel",sobelx)
cv2.imshow("sobel edge",edge)
cv2.imshow("img",img)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()