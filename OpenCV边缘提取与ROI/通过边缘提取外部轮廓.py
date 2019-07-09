import cv2
import numpy as np

# Canny的输入图像是单通道8位图像，灰度图像
def onChange(pos):
    global gray,threshold1,threshold2
    img2 = img.copy()
    threshold1 = cv2.getTrackbarPos("threshold1","origin")
    threshold2 = cv2.getTrackbarPos("threshold2", "origin")
    kernel = np.ones((10,10),np.uint8)
    morph = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel=kernel)
    cannyimg = cv2.Canny(morph,threshold1,threshold2)
    image, contours, hierarchy = cv2.findContours(cannyimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    max_region= 0
    max_rect = 0,0,0,0
    for obj in contours:
        x, y, w, h = cv2.boundingRect(obj)
        region = w * h
        if region > max_region:
            max_region = region
            max_rect = [x, y, w, h]
    x, y, w, h = max_rect
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("findcontour",image)
    cv2.imshow("origin", img2)
img = cv2.imread("chessimg/laser01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold1 = 100
threshold2 = 300
cv2.namedWindow("origin",0)
cv2.createTrackbar("threshold1","origin",0,500,onChange)
cv2.createTrackbar("threshold2","origin",0,500,onChange)
cv2.imshow("gray",gray)
cv2.waitKey(0)

