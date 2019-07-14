# 利用霍夫变换求取直线的端点
# 通过两个端点来直线的一般式方程：f(x) = ax+by+c
# 两条直线的交点：a1x + b1y + c1 = a1x + b2y + c2
# x0 = (b1c2-b2c1)/D
# y0 = (a2c1-a1c2)/D
# D = a1*b2-a2*b1
import cv2
import numpy as np

def ComputeK(line):
    line = line[0]
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    k = (y2-y1)/(x2-x1)
    return k
def get_3idK(lines):
    K = np.zeros(len(lines))
    for i,line in enumerate(lines):
        k = ComputeK(line)
        K[i] = k
    maxK= np.argmax(K)
    minK = np.argmin(K)
    midK = np.argmin(np.abs(K))
    #return [[maxK,K[maxK]],[midK,K[midK]],[minK,K[minK]]]
    return [maxK,midK,minK]
def getPara(line):
    x1=  line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    A = y2-y1
    B = x1-x2
    C = x2*y1-x1*y2
    return [A,B,C]
def getCross(para1,para2):
    D = para1[0]*para2[1]-para2[0]*para1[1]
    x0 = (para1[1]*para2[2]-para2[1]*para1[2])/D
    y0 = (para2[0]*para1[2]-para1[0]*para2[2])/D
    return [x0,y0]
def getLines(gray):
    minLineLength = 250
    maxLineGap = 10
    lines = cv2.HoughLinesP(gray, 1.0, np.pi / 180, 250, minLineLength=minLineLength, maxLineGap=maxLineGap)
    id3 = get_3idK(lines)
    return lines[id3]

def drawLine(lines,img):
    color = [[255,0,0],[0,255,0],[0,0,255]]
    for i in range(len(lines)):
        line = lines[i][0]
        line1 = lines[i][0]
        line2 = lines[(i+1)%3][0]
        para1 = getPara(line1)
        if i==0:
            cv2.line(img, (line[0], line[1]), (line[2]+int((-para1[1]*100/para1[0])), line[3]+100), color[i],2)
        elif i == 2:
            cv2.line(img, (line[0]+ int((-para1[1] * 100 / para1[0])), line[1]+100), (line[2] , line[3]), color[i],2)
        else:
            cv2.line(img, (line[0]-700, line[1]+int((para1[0] * 700 / para1[1]))), (line[2], line[3]), color[i],2)
        para2 = getPara(line2)
        p = getCross(para1,para2)
        print(p)
        cv2.putText(img, "[{:.3f},{:.3f}]".format(p[0], p[1]), (int(p[0])-10,int(p[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255,255), 2, lineType=cv2.LINE_8)
        cv2.circle(img,(int(p[0]),int(p[1])),4,[255,0,255],3 )
    #cv2.circle(img, (int(p[1]), int(p[1])), 2, [0, 255, 0], -1)

img = cv2.imread("test01.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
lines = getLines(gray)
print(lines,"####")
newimg = np.zeros(gray.shape,dtype=np.uint8)
drawLine(lines,img)
cv2.imshow("origin",img)
cv2.imshow("newimg",newimg)
cv2.waitKey(0)


