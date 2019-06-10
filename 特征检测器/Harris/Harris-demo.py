import cv2
import numpy as np

img_org = cv2.imread('demo02.jpg')
img = cv2.resize(img_org,(600,800))
img_com = cv2.resize(img_org,(300,400))


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray  = np.float32(gray)

dst = cv2.cornerHarris(gray,2,23,0.04)
print("600*800",np.sum(dst>0.01*dst.max()))

gray_com = cv2.cvtColor(img_com,cv2.COLOR_BGR2GRAY)
gray_com = np.float32(gray_com)
dst_com = cv2.cornerHarris(gray_com,2,23,0.04)
print("300*400",np.sum(dst_com>0.01*dst_com.max()))

print(dst.shape)
print(dst>0.01*dst.max())
img[dst>0.01*dst.max()] = [0,255,0]
img_com[dst_com>0.01*dst_com.max()] = [0,0,255]
while True:
    cv2.imshow('corner',img)
    cv2.imshow('corner_com',img_com)
    if cv2.waitKey(2000) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()