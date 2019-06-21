import cv2
import numpy as np
def GravityPlus(img):
    row, col, chanel = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    textimg= np.zeros((row,col))
    points = []
    newimage = np.zeros((row, col), np.uint8)
    for i in range(col):
        Pmax = np.max(gray[:, i])
        Pmin  = np.min(gray[:, i])
        if Pmax==Pmin:
            continue
        #print("Pmax",Pmax)
        pos = np.argwhere(gray[:,i]>=(Pmax-5))
        #print("pos",pos)
        length = len(pos)
        sum_top,sum_down =0.0,0.0
        if pos[-1]-pos[0]==length-1:
            #print("good cols",i)
            for p in pos:
                sum_top += p*gray[p,i]
                sum_down+=gray[p,i]
            Prow = sum_top/sum_down
            points.append([Prow[0],i])
    for p in points:
        #print(p)
        pr,pc = map(int,p)
        textimg[pr,pc] = 255
        newimage[pr,pc] = 255
        img[pr,pc,:] = [0,255,0]
    cv2.namedWindow("Plus_origin",0)
    cv2.namedWindow("Plus_centerLine",0)
    cv2.imshow("Plus_origin",img)
    np.savetxt("center-v.txt",textimg,fmt="%.3d")
    cv2.imwrite("center-line.jpg",textimg)
    cv2.imshow("Plus_centerLine",newimage)
    return points
img = cv2.imread("center-v.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
points = np.array(GravityPlus(img))
def findFeatur(points):
    Fpoints = []
    rootindex = np.argmax(points[:,0])
    rootPoint=  points[rootindex].astype(int)
    print(rootPoint)
    Fpoints.append(rootPoint)


    return Fpoints

Fpoints = findFeatur(points)
cv2.circle(img,(Fpoints[0][1],Fpoints[0][0]),2,[0,0,255] ,-1)
cv2.imshow("origin",img)
cv2.waitKey(0)
