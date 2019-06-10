import cv2

img = cv2.imread("14.jpg")

print("start!")
cv2.imshow("winname",img)
a = cv2.waitKey(3000)
print("after 3000 ")
cv2.waitKey(1)
print("1 ms")
cv2.waitKey(5000)
print("5000 ")