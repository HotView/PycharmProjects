import numpy as np
import matplotlib.pyplot as plt
FLOAT_MAX =1e100
def load_data(file_path):
    """
    :param file_path:
    :return:
    """
    f = open(file_path)
    data = []
    for line in f.readlines():
        row = []
        lines = line.strip().split("\t")
        for x in lines:
            row.append(float(x))
        data.append(row)
    f.close()
    return np.mat(data)
def distance(VecA,VecB):
    dist =(VecA-VecB)*(VecA-VecB).T
    return dist
def neraest(point,cluster_centers):
    """

    :param point:
    :param cluster_centers:
    :return:
    """
    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]
    for i in range(m):
        d = distance(point,cluster_centers[i])
        if min_dist>d:
            min_dist =d
    return min_dist
def get_centroids(points,k):
    """
    :param points:
    :param k:
    :return:
    """
    m,n = np.shape(points)
    cluster_centers=np.mat(np.zeros((k,n)))
    #print(cluster_centers.shape,"#####")
    index = np.random.randint(0,m)
    cluster_centers[0] = np.copy(points[index])
    d = [0.0 for i in range(m)]

    for i in range(1,k):
        sum_all = 0
        for j in range(m):
            d[j] = neraest(points[j],cluster_centers[0:i,])
            sum_all+=d[j]
        sum_all*=np.random.random()
        for v,di in enumerate(d):
            sum_all-=di
            if sum_all>0:
                continue
            #print(i,v,"---")
            #print(cluster_centers.shape)
            cluster_centers[i] = np.copy(points[v,])
            break
    return cluster_centers
def kmeanspp(data,k,centroids):
    """
    :param data:
    :param k:
    :param centroids:
    :return:
    """
    m,n  = np.shape(data)
    subCenter = np.mat(np.zeros((m,2)))
    change = True
    while change==True:
        change  =False
        for i in range(m):
            minDist = np.inf
            minIndex = 0
            for j in range(k):
                dist = distance(data[i],centroids[j])
                if minDist>dist:
                    minDist = dist
                    minIndex = j
            if subCenter[i,1]!= minDist:
                change = True
                subCenter[i] = np.mat((minIndex,minDist))
        for j in range(k):
            sum_all = np.mat(np.zeros((1,n)))
            r = 0
            for i in range(m):
                if subCenter[i,0] == j:
                    sum_all +=data[i]
                    r =r+1
            for z in range(n):
                try:
                    centroids[j,z] = sum_all[0,z]/r
                except:
                    print("r in zero")
    return centroids,subCenter
def save_result(file_name,source):
    m,n = np.shape(source)
    f = open(file_name,"w")
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i,j]))
        f.write("\t".join(tmp)+"\n")
    f.close()

if __name__ == '__main__':
    k = 4
    file_path = "data.txt"
    data = load_data(file_path)
    centroids = get_centroids(data,k)
    centroids,subCenter= kmeanspp(data,k,centroids)
    plt.figure()
    for index, x in enumerate(data):
        if int(subCenter[index, 0]) == 0:
            plt.scatter(data[index, 0], data[index, 1], c='b')
        elif int(subCenter[index, 0]) == 1:
            plt.scatter(data[index, 0], data[index, 1], c='r')
        elif int(subCenter[index, 0]) == 2:
            plt.scatter(data[index, 0], data[index, 1], c='g')
        else:
            plt.scatter(data[index, 0], data[index, 1], c='y')
    plt.show()
    save_result("sub_pp",subCenter)
    save_result("center_pp",centroids)