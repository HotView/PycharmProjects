#测量面积，几何矩等等
import cv2
import numpy as np
def measure_object(image):
    image =  cv2.GaussianBlur(image,(5,5),0)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    dst = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    print("threshold value",ret)
    cv2.imshow("binary image",binary)
    outimage,contours,hireachy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        Area = cv2.contourArea(contour)
        print("contour-area:",Area)
        x,y,w,h =cv2.boundingRect(contour)
        rate = min(w,h)/max(w,h)
        print("rectangle rate:",rate)
        mm = cv2.moments(contour)
        print(mm)
        if mm['m00']:
            cx = mm["m10"]/mm['m00']
            cy = mm["m01"]/mm['m00']
            cv2.circle(image,(np.int(cx),np.int(cy)),3,(0,255,255),-1)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
            approx_ploy = cv2.approxPolyDP(contour,4,)
            print(approx_ploy.shape)
            if approx_ploy.shape[0]==3:
                cv2.drawContours(dst,contours,i,(0,255,0),2)
    cv2.imshow("measure_contours",dst)
image = cv2.imread("1.jpg")
measure_object(image)
cv2.waitKey()