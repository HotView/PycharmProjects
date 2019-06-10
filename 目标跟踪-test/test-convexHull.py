import cv2

img = cv2.imread("yibiaopan.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("origin",img)
cv2.imshow("gray",gray)
binary = cv2.Canny(img,100,200)
cv2.imshow("edges",binary)
cv2.findContours

cv2.imread
cv2.waitKey(0)
cv2.Canny

