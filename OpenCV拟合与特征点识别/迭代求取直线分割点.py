import cv2
import numpy as np
from OpenCV相机标定.FitLine import getFitline
from OpenCV中心线提取.ExtCenterPts import ExtremeCenter


def Crosspoint(line1,line2):
    k1, b1 = line1
    k2, b2 = line2
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y
img = cv2.imread("laser.jpg")
img = img[:,50:,:]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

points = ExtremeCenter(gray)
error = 0.002
sort_points = sorted(points,key=lambda x:x[0])
#print(points)
mididex = int(len(sort_points)/2)
while(1):
    left_points = np.array(sort_points[0:mididex])
    right_points = np.array(sort_points[mididex:])
    print("#############################")
    print(left_points)
    line1 = getFitline(left_points)
    line2 = getFitline(right_points)
    crosspoint = Crosspoint(line1,line2)
    dists = np.sum(np.square(np.array(sort_points)-crosspoint),axis=1)
    minindex = np.argmin(dists)
    if mididex==minindex:
        break
    mididex = minindex
    # error_x = sort_points[mididex][0] - crosspoint[0]
    # error_y= sort_points[mididex][1] - crosspoint[1]
    # print(error_x,error_y)
    # if error_x<error and error_y<error:
    #     break
    #
    print("index", mididex)
k1,b1 = line1
k2,b2 = line2
cv2.line(img,tuple(map(int,crosspoint)),(0,int(b1)),[255,0,0],1)
cv2.line(img,tuple(map(int,crosspoint)),(1000,int(1000*k2+b2)),[255,0,0],1)
cv2.putText(img,"[{:.3f},{:.3f}]".format(crosspoint[0],crosspoint[1]),tuple(sort_points[mididex]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255),1,lineType=cv2.LINE_8)
cv2.circle(img,tuple(sort_points[mididex]),4,[0,255,0])
cv2.namedWindow("img",0)
cv2.imshow("img",img)
cv2.waitKey(0)



