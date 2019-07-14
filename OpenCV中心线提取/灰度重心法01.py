import cv2
import numpy as np

def GravityCen(img):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = []
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        pos = np.argmax(gray[:, i])
        Pmax = gray[pos, i]
        Pmin  = np.min(gray[:, i])
        if Pmax==Pmin:
            continue
        length = 2
        sum =0.0
        down = 0.0
        for j in range(-2,3):
            colp = pos+j
            sum += colp*gray[colp,i]
            down +=gray[colp,i]
        Prow = sum/down
        points.append([Prow,i])
    for p in points:
        #print(p)
        pr,pc = map(int,p)
        newimage[pr,pc] = 255
        img[pr,pc,:] = [0,255,0]
    cv2.namedWindow("Plus_origin",0)
    cv2.namedWindow("Plus_centerLine",0)
    cv2.imshow("Plus_origin",img)
    cv2.imshow("Plus_centerLine",newimage)
    return points
img = cv2.imread("image/laser-v02.jpg")
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#np.savetxt("laser-v02.txt",gray,fmt="%.3d")
#rows,cols = gray.shape
blur = cv2.blur(img,(1,10))
points = GravityCen(blur)
#cv2.imshow("gray",gray)
cv2.imshow("blur",blur)
cv2.waitKey(0)

