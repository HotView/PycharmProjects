import cv2
import numpy as np
def onChange(val = 0):
    img = cv2.imread("laser.jpg")
    bin_thresh = cv2.getTrackbarPos("Binary_threshold","mainwindow")
    hough_thresh = cv2.getTrackbarPos("Hough_threshold","mainwindow")
    retval, threimg = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLines(threimg, 1, np.pi / 180, hough_thresh)
    # print(type(lines))
    lines_re = np.reshape(lines, (-1, 2))
    cv2.imshow("thresh", threimg)
    for rho, theta in lines_re:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("hough", img)
cv2.namedWindow("mainwindow",0)
cv2.createTrackbar("Binary_threshold","mainwindow",0,255,onChange)
cv2.createTrackbar("Hough_threshold","mainwindow",0,255,onChange)
img = cv2.imread("laser.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("mainwindow",img)
cv2.waitKey(0)
