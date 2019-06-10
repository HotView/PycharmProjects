import cv2
def contours_demo(image):
    dst = cv2.GaussianBlur(image,(3,3),0)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    cv2.imshow("binary image",binary)

    cloneimage,contours,heriachy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for k,contour in enumerate(contours):
        cv2.drawContours(image,contours,k,(0,0,255),-1)#轮廓的线宽为-1的话，填充整个轮廓。
        print(k)
    cv2.imshow("detection contours",image)
image = cv2.imread("2.jpg")
contours_demo(image)
help(cv2.HoughLines)
cv2.waitKey()