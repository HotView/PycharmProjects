import numpy as np
import math
MIN_DISTANCE = 0.000001
def load_data(path,feature_num = 2):
    """
    导入数据
    :param path:文件的存储位置
    :param feature_num: 特征的个数
    :return: 特征
    """
    f = open(path)
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        if len(lines)!= feature_num:
            continue
        for i in range(feature_num):
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    f.close()
    return data
def gaussian_kernel(distance,bandwidth):
    """
    :param distance:
    :param bandwidth:
    :return:
    """
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m,1)))
    for i in range(m):
        right[i,0] = (-0.5*distance[i]*distance[i].T)\
                     /(bandwidth*bandwidth)
        right[i,0] = np.exp(right[i,0])
    left  = 1/(bandwidth*math.sqrt(2*math.pi))
    gaussian_val = left*right
    return gaussian_val
def eculidean_dist(pointA,pointB):
    """
    :param pointA:
    :param pointB:
    :return:
    """
    #print(pointA,"########")
    #print(pointA.shape)
    total = (pointA-pointB)*((pointA-pointB).T)
    #print(total,"_________")
    return math.sqrt(total)
def shift_point(point,points,kernel_bandwidth):
    """计算均值漂移点
    :param point:
    :param points:
    :param kernel_bandwidth:
    :return:
    """
    points = np.mat(points)
    m = np.shape(points)[0]# 样本的个数
    #计算距离
    point_distances = np.mat(np.zeros((m,1)))
    for i in range(m):
        point_distances[i,0] =eculidean_dist(point,points[i])
    #计算高斯核函数
    point_weights = gaussian_kernel(point_distances,kernel_bandwidth)
    # 计算分母
    all_sum = 0.0
    for i in range(m):
        all_sum+=point_weights[i]
    #均值偏移
    point_shifted = point_weights.T*points/all_sum# 1*m*m*n = 1*n
    return point_shifted  # 一个点的各个维度的一个漂移值
def group_points(mean_shift_points):
    """
    计算所属的类别
    :param mean_shift_points:
    :return:
    """
    group_assignment = []
    m,n = np.shape(mean_shift_points)
    index= 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f"%mean_shift_points[i,j])))
            print(item)
        item_1 = "_".join(item)
        print(item_1)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index = index+1
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f"%mean_shift_points[i,j])))
        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])
    return group_assignment
def train_mean_shift(points,kernel_bandwidth =2):
    """训练Meanshift模型
    :param points:
    :param kernel_bandwidth:
    :return:
    """
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0#训练的次数
    m = np.shape(mean_shift_points)[0]
    need_shift = [True]*m
    #计算均值漂移向量
    while max_min_dist>MIN_DISTANCE:
        max_min_dist = 0
        iteration +=1
        print("\iteration:"+str(iteration))
        for i in range(0,m):
            # 判断每一个样本点是否需要计算偏移均值
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            #对样本点进行漂移
            p_new = shift_point(p_new,points,kernel_bandwidth)
            # 计算该点与漂移后的点之间的距离
            dist = eculidean_dist(p_new,p_new_start)

            if dist>max_min_dist:
                max_min_dist =dist
            if dist<MIN_DISTANCE:#不需要移动
                need_shift[i] = False
            mean_shift_points[i] = p_new
    #计算最终的group
    group = group_points(mean_shift_points)
    return np.mat(points),mean_shift_points,group
def save_result(path_name,source):
    m,n = np.shape(source)
    f = open(path_name,"w")
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i,j]))
        f.write("\t".join(tmp)+"\n")
    f.close()
if __name__ == '__main__':
    data = load_data("data",2)
    points,shift_points,cluster = train_mean_shift(data,2)
    save_result("sub",np.mat(cluster))
    save_result("center",shift_points)
