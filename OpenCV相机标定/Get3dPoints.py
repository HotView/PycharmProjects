import cv2
import numpy as np
from scipy.optimize import leastsq
def distance(vec1,vec2):
    #print(type(vec1))
    #print(type(vec2))
    dist = np.sqrt(np.sum(np.square(vec1-vec2)))
    return dist
def f_1(p,y,x):
    k,b = p
    return y-(k*x+b)
def fit_line(line_points):
    x0 = line_points[:,0]
    y0 = line_points[:,1]
    p0 = [1.,0.0]
    k,b  = leastsq(f_1, p0,args=(y0,x0))[0]
    return k,b
def cross_point(line1,line2):
    k1 ,b1 = line1
    k2 ,b2 = line2
    x = (b2-b1)/(k1-k2)
    y = k1*x+b1
    return x,y
def point_3_index(cross_point,line_points):
    cross_point = np.array(cross_point)
    distxy = np.square(line_points-cross_point)
    dist = np.sum(distxy,axis=1)
    index = np.argsort(dist)[:3]
    A = line_points[index[2]]
    AB = distance(A,line_points[index[0]])
    AC = distance(A,line_points[index[1]])
    if AB>AC:
        tmp = index[0]
        index[0] = index[1]
        index[1] = tmp
    return index
def get3dPoints(laserline,corners,objp):
    points_3d = []
    points_neigh = []
    crosspoints = []
    close_points = []
    for i,line_points in enumerate(corners):
        line_i = fit_line(line_points)
        k1,b1 = line_i
        crosspoint = np.array(cross_point(line_i,laserline))
        #print(crosspoint)
        crosspoints.append(crosspoint)
        index_neighbor = point_3_index(crosspoint,line_points)
        #print("index_neighbor",index_neighbor)
        A = line_points[index_neighbor[2]]
        B = line_points[index_neighbor[0]]
        C = line_points[index_neighbor[1]]
        points_neigh.append([A,B,C])
        AD = distance(A, crosspoint)
        BD = distance(B, crosspoint)
        AC = distance(A, C)
        BC = distance(B, C)
        K = (AD / BD) / (AC / BC)
        l = 1
        solution_x = l / (2 * K - 1)
        B_3d = objp[i][index_neighbor[0]] * l
        # 角点生成的顺序是按照X轴递增的顺序生成的，靶标坐标系中对应点的X分量也是递增的
        # 所以可以根据A,C点在角点序列中的位置来判断，AC方向的分量在坐标轴中是递增的还是递减的
        if index_neighbor[2]>index_neighbor[1]:
            point_3d = [B_3d[0]-solution_x,B_3d[1],B_3d[2]]
        else:
            point_3d = [B_3d[0]+solution_x,B_3d[1],B_3d[2]]
        points_3d.append(point_3d)
        close_points.append(B)
    return points_3d,points_neigh,crosspoints,close_points