import cv2

def nothing(val = 0):
    kersize = cv2.getTrackbarPos("ker_size", "origin")
    kersize = kersize * 2 + 1
    sigmax = cv2.getTrackbarPos("sigmaX", "origin")
    threshold = cv2.getTrackbarPos("threshold", "origin")
    blur = cv2.GaussianBlur(img, (kersize, kersize), sigmax)
    gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thres_img = cv2.threshold(gray_blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("blur", blur)
    cv2.imshow("thresh", thres_img)

img = cv2.imread("laser.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img,100,200)
cv2.namedWindow("origin")
cv2.createTrackbar("ker_size","origin",0,15,nothing)
cv2.createTrackbar("sigmaX","origin",0,20,nothing)
cv2.createTrackbar("threshold","origin",0,255,nothing)
cv2.imshow("origin",img)
cv2.imshow("edges",edges)
cv2.waitKey(0)
cv2.destroyAllWindows()