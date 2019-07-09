import cv2
import numpy as np
from scipy.optimize import curve_fit


def distance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist
def f_1(x,A,B):
    return A*x+B
def fit_line(line_points):
    x0 = line_points[:,0]
    y0 = line_points[:,1]
    A1, B1 = curve_fit(f_1, x0, y0)[0]
    return A1,B1
def corss_point(line1,line2):
    k1 = line1[0]
    b1 = line1[1]
    k2 = line2[0]
    b2 = line2[1]
    x = (b2-b1)/(k1-k2)
    y = k1*x+b1
    return x,y
def point_3_index(cross_point,points):
    cross_point = np.array(cross_point)
    #print(cross_point,"#")
    #print(points)
    t = np.square(points-cross_point)
    print(t.shape)
    print(t)
    distxy = np.square(points-cross_point)
    #print(distxy)
    dist = np.sum(distxy,axis=1)
    #print(dist)
    index = np.argsort(dist)[:3]
    A = line_points[index[2]]
    AB = distance(A,line_points[index[0]])
    AC = distance(A,line_points[index[1]])
    if AB>AC:
        tmp = index[0]
        index[0] = index[1]
        index[1] = tmp
    return index

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
objp = np.zeros((6*7,3),np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
objp = objp.reshape((6,7,3))
print("3D point",objp)
fname = "image/left01.jpg"
img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,corners  =cv2.findChessboardCorners(gray,(7,6),None)
if ret:
    print(corners.shape)
    for i in corners:
        print(i)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #print("corners2", corners2)
    # 绘制和展示角点,按照颜色来进行划分（7,6），6种颜色
    corners3= corners2.reshape((6,7,2))
    test_point = np.array([[300, 0], [400, 800]])
    line_laser = fit_line(test_point)
    for j,line_points in enumerate(corners3):
        line_i = fit_line(line_points)
        A1, B1 = line_i
        #-----------------------------
        x1= np.arange(0,1500,500)
        y1 = A1*x1+B1
        #--------------------------------
        cross_point = corss_point(line_i,line_laser)
        #print(cross_point)
        x,y = int(cross_point[0]),int(cross_point[1])
        index_i = point_3_index(cross_point,line_points)
        #print(index_i)
        A = line_points[index_i[2]]
        B = line_points[index_i[0]]
        C = line_points[index_i[1]]
        AD = distance(A,cross_point)
        BD = distance(B,cross_point)
        AC = distance(A,C)
        BC = distance(B,C)
        K = (AD/BD)/(AC/BC)
        #print(K)
        l = 50
        solution_x  =  l/(2*K-1)

        B_3d = objp[j][index_i[0]]*50
        if A[0]>C[0]:
            point_3d = [B_3d[0]+solution_x,B_3d[1],B_3d[2]]
        else:
            point_3d = [B_3d[0]-solution_x,B_3d[1],B_3d[2]]
        print("---")
        print(B_3d)
        print(point_3d)
        print("---")

        #print("distance",AD,BD,AC,BC)
        cv2.circle(img,(x,y),3,[0,255,0],3)
        for i in index_i:
            x = int(line_points[i][0])
            y = int(line_points[i][1])
            cv2.circle(img, (x,y), 3, [255, 0, 0], 2)
        #cross_point = cross_point
        cv2.line(img,(x1[0],int(y1[0])),(x1[-1],int(y1[-1])),[0,0,255],1)

    cv2.line(img, (300, 0), (400, 800), [0, 0, 255], 2)
    #img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv2.namedWindow(fname,0)
    cv2.imshow(fname, img)
cv2.waitKey(0)
