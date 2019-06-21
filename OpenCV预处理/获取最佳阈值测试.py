import cv2
def onChage(pos):
    retval,threimg = cv2.threshold(gray,pos,255,cv2.THRESH_BINARY)
    cv2.imshow("thresh",threimg)
cv2.namedWindow("thresh",0)
img_laser = cv2.imread("laser01.jpg")
cv2.imshow("origin",img_laser)
gray = cv2.cvtColor(img_laser,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.createTrackbar("bar","thresh",0,255,onChage)

cv2.waitKey(0)