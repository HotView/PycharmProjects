import cv2
import numpy as np
def onChange(val = 0):
    img = cv2.imread("laser.jpg")
    bin_thresh = cv2.getTrackbarPos("Binary_threshold","mainwindow")
    hough_thresh = cv2.getTrackbarPos("Hough_threshold","mainwindow")
    retval, threimg = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(threimg, 1, np.pi / 180, hough_thresh,minLineLength=30,maxLineGap=10)
    # print(type(lines))
    cv2.imshow("thresh", threimg)
    lines_re = np.reshape(lines, (-1, 4))
    for line in lines_re:
        x0,y0,x1,y1 = line
        cv2.line(img,(x0,y0),(x1,y1),(0,255,0))
    cv2.imshow("hough", img)
cv2.namedWindow("mainwindow",0)
cv2.createTrackbar("Binary_threshold","mainwindow",0,255,onChange)
cv2.createTrackbar("Hough_threshold","mainwindow",0,255,onChange)
img = cv2.imread("laser.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("mainwindow",img)
cv2.waitKey(0)
