# 寻找最短权值（快）的路径
# 找出最便宜的节点
# 更新这个节点的邻居的开销
# 重复上述过程，直到每个节点都这样做
# 数据结构，边和顶点，有一个dis数组 和 标记数组book，很关键，不断更新值
one = input()
num,edge = list(map(int,one.split()))
k = 0
inf = 10000
e = [[inf for i in range(num)]for j in range(num)]
dis = [inf for i in range(num)]
book = [0 for i in range(num)]
book[k] = 1
for i in range(edge):
    m,n,value= list(map(int,input().split()))
    e[m-1][n-1] =value
print(e)
for i in range(num):
    dis[i] = e[k][i]
dis[k] = 0
for i in range(num):
    minvalue = inf
    for v in range(num):
        if book[v]==0 and dis[v]<minvalue:
            minvalue  = dis[v]
            index = v
    print("------")
    print(minvalue)
    print(index)
    book[index] = 1
    # 判断最近的节点的临点的值的变化
    for j in range(num):
        if e[index][j]<inf:
            if dis[j]>dis[index]+e[index][j]:
                dis[j] = dis[index]+e[index][j]
print(dis)

