import cv2

img1 = cv2.imread('01.jpg')
img2 = cv2.imread('02.jpg')

img1_re = cv2.resize(img1,(800,600))
img2_re = cv2.resize(img2,(800,600))

cv2.imwrite('01_resize.jpg',img1_re)
cv2.imwrite('02_resize.jpg',img2_re)