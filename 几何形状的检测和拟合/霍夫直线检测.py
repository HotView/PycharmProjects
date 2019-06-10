import numpy as np
import cv2
import math
def HTLine(image,stepTheta= 1,stepRho= 1):
    rows,cols = image.shape
    # 图像中可能出现的最大垂线的长度
    L = round(math.sqrt(pow(rows-1,2.0)+pow(cols-1,2.0)))+1
    # 初始化投票器
    numtheta = int(180.0/stepTheta)
    numRho = int(2*L/stepRho+1)
    accumlator =  np.zeros((numRho,numtheta),np.int32)
    # 建立字典
    accuDict = {}
    for k1 in range(numRho):
        for k2 in range(numtheta):
            accuDict[(k1,k2)] = []
    # 投票计数
    for y in range(rows):
        for x in range(cols):
            if(image[y][x] == 255):#只对边缘点做霍夫变换
                for m in range(numtheta):
                    rho = x*math.cos(stepTheta*m/180.0*math.pi)+y*math.sin(stepTheta*m/180.0*math.pi)
                    # 计算投票哪一个区域
                    n = int(round(rho+L)/stepRho)
                    # 投票加1
                    accumlator[n,m]+=1
                    accuDict[(n,m)].append((y,x))
    return accumlator,accuDict

print(round(3.40))
print(round(3.50))
print(round(3.602157))
