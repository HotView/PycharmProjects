import cv2
import numpy as np
points = []
"""
按行坐标排序，取前后k个数值，然后拟合这些点进行拟合直线
按列坐标排序，选取列坐标最大的点为凹点，即凹槽最低点，记为C点
以C点为基准，设定一个范围t，取列坐标在[yh-t,yh]区间的点，然后以xh为分界点分为两部分
分别对两部分进行直线拟合，然后求出这两条直线和主直线的交点
"""
pointsSortX = sorted(points,key=lambda x:x[1])
pointsSortY = sorted(points,key=lambda x:x[0])
