import cv2
import numpy as np

# Canny的输入图像是单通道8位图像，灰度图像
def onChange(pos):
    global gray,threshold1
    img2 = img.copy()
    threshold = cv2.getTrackbarPos("threshold","origin")
    ret,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    #---------------------
    kernel = np.ones((2, 25), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #---------------------
    image, contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for obj in contours:
        x,y,w,h = cv2.boundingRect(obj)
        #vertices = cv2.boxPoints(rotatedRect)
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
        #cv2.rectangle(morph,(x,y),(x+w,y+h), 255, 2)
    print(len(contours))
    cv2.imshow("findcontour",image)
    cv2.imshow("morph", morph)
    cv2.imshow("thresh",thresh)
    cv2.imshow("color_image", img2)
img = cv2.imread("chessimg/laser01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold1 = 0
cv2.namedWindow("origin",0)
cv2.createTrackbar("threshold","origin",0,255,onChange)
cv2.imshow("origin",img)
cv2.waitKey(0)

