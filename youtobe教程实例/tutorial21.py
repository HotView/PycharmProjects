import cv2
import numpy as np
def line_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,100,apertureSize=3)#边缘检测时，窗口指定的大小
    lines = cv2.HoughLines(edges,1,np.pi/180,120)
    for line in lines:
        print(line)
        print('#########')
        rho,theta = line[0]
        a = np.cos(theta)
        b =np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000 * (-b))
        y2 = int(y0-1000 * (a))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('image_line',image)
def line_detectionP(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize=3)  # 边缘检测时，窗口指定的大小
    lines = cv2.HoughLinesP(edges,1,np.pi/180,120,minLineLength=250,maxLineGap=5)#minLineLength和maxLineGap很重要在图像处理过程中！
    for line in lines:
        print(type(line))
        x1,y1,x2,y2 = line[0]#线段的起始点和终止点
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('image_line_HoughsPPPP', image)


image = cv2.imread("4.jpg")
line_detection(image)
line_detectionP(image)
help(cv2.HoughLines)
cv2.waitKey()