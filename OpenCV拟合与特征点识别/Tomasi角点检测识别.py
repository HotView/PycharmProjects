import cv2
import numpy as np

def nothing(val):
    newimg = img.copy()
    maxcorners = cv2.getTrackbarPos("maxcorners","origin")
    qualityLevel = cv2.getTrackbarPos("qualityLevel","origin")/100.0
    minDistance = cv2.getTrackbarPos("minDistance","origin")
    points1 = cv2.goodFeaturesToTrack(detector,maxCorners=maxcorners,qualityLevel=qualityLevel,minDistance=minDistance)
    print(points1)
    for point in points1:
        print("point",point[0])
        cv2.putText(newimg, "[{:.2f},{:.2f}]".format(point[0][0], point[0][1]), (int(point[0][0]-20),int(point[0][1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, lineType=cv2.LINE_8)
        cv2.circle(newimg,(point[0][0],point[0][1]),5,[0,0,255],-1 )
    cv2.imshow("res",newimg)
img = cv2.imread("center-v1.jpg")
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Ix = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
Iy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
Ixx = cv2.Scharr(Ix, cv2.CV_32F, 1, 0)
Ixy = cv2.Scharr(Ix, cv2.CV_32F, 0, 1)
Iyx = cv2.Scharr(Iy, cv2.CV_32F, 1, 0)
Iyy = cv2.Scharr(Iy, cv2.CV_32F, 0, 1)
detector = (cv2.GaussianBlur(Ixy, (5, 5), 0) *
            cv2.GaussianBlur(Iyx, (5, 5), 0) -
            cv2.GaussianBlur(Ixx, (5, 5), 0) *
            cv2.GaussianBlur(Iyy, (5, 5), 0))
cv2.namedWindow("origin")
cv2.createTrackbar("maxcorners","origin",0,10,nothing)
cv2.createTrackbar("qualityLevel","origin",2,10,nothing)
cv2.createTrackbar("minDistance","origin",5,10,nothing)

cv2.imshow("origin",gray)
cv2.waitKey(0)