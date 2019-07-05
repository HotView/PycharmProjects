import numpy as np
import matplotlib.pyplot as plt
def distance(VecA,VecB):
    """
    计算两个向量之间的距离
    :param VecA: A点坐标
    :param VecB: B点坐标
    :return: A点与B点距离的平方
    """
    dist = (VecA-VecB)*(VecA-VecB).T
    return dist[0,0]
def load_data(file_path):
    """
    :param file_path:文件存储的位置
    :return: 数据
    """
    f  = open(file_path)
    data = []
    for line in f.readlines():
        row= []
        lines = line.strip().split("\t")
        for x in lines:
            row.append(float(x))
        data.append(row)
    f.close()
    return np.mat(data)
def randCent(data,k):
    """
    随机初始化聚类中心
    :param data:训练数据
    :param k:类别个数
    :return:聚类中心
    """
    n = np.shape(data)[1]# data dim
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = np.min(data[:,j])
        rangeJ = np.max(data[:,j])-minJ
        centroids[:,j] = minJ*np.mat(np.ones((k,1)))\
        +np.random.rand(k,1)*rangeJ
    return centroids
def Kmeans(data,k,centroids):
    """
    compute the solution
    :param data:训练数据
    :param k: 类别个数
    :param centroids: 随机初始化的聚类中心
    :return :centroids：训练完成的聚类中心
            :subCenter：每一个样本所属类别
    """
    m,n  = np.shape(data)
    subCenter = np.mat(np.zeros((m,2)))# 初始化每个样本所属的类别
    change = True
    while change==True:
        change = False
        for i in range(m):
            minDist = np.inf
            minIndex = 0
            for j in range(k):
                #计算i和每个聚类中心之间的距离,取最小的那一类！
                dist = distance(data[i,],centroids[j,])
                if dist<minDist:
                    minDist = dist
                    minIndex = j
            if subCenter[i,0] != minIndex:
                change = True
                subCenter[i,] = np.mat([minIndex,minDist])
            else:
                subCenter[i,] = np.mat([minIndex,minDist])
        #重新计算聚类中心
        for j in range(k):
            #每类中所有样本数据的每一维的和
            sum_all = np.mat(np.zeros((1,n)))
            r = 0#每个类别中样本的个数
            for i in range(m):
                if subCenter[i,0]==j:
                    sum_all +=data[i,]
                    r+=1
            for z in range(n):
                try:
                    centroids[j,z] = sum_all[0,z]/r
                except :
                    print("r is zero")
    return centroids,subCenter
def save_result(file_name,source):
    """
    保存source中的结果到file_name文件中
    :param file_name:
    :param source:
    :return:
    """
    m,n = np.shape(source)
    with open(file_name,"w") as f:
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i,j]))
            f.write("\t".join(tmp)+"\n")
    #f.close()
if __name__ == '__main__':
    k = 4
    file_path = "data.txt"
    print("--------1.load_data-----")
    data = load_data(file_path)
    print(data.shape)
    print("--------1.random center-----")
    centroid = randCent(data,k)
    print("--------3.k-means-----")
    centroids,subCenter = Kmeans(data,k,centroid)
    print("--------4.save_solute-----")
    plt.figure()
    for index,x in enumerate(data):
        if int(subCenter[index,0])==0:
            plt.scatter(data[index,0],data[index,1],c = 'b')
        elif int(subCenter[index, 0]) == 1:
            plt.scatter(data[index, 0], data[index, 1], c='r')
        elif int(subCenter[index, 0]) == 2:
            plt.scatter(data[index, 0], data[index, 1], c='g')
        else :
            plt.scatter(data[index, 0], data[index, 1], c='y')
    plt.show()
    save_result("sub",subCenter)
    print("--------5.SAVE centroids-----")
    save_result("center",centroids)
