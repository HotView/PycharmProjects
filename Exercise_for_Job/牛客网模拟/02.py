#01背包问题
#m个物品
#每个物品价值为x，耗费空间为y
#总共有空间时n+1
#求最小
vol = []
money = []
weight  =[]
line = list(map(int,input().split()))
n = line[0]
m = line[1]
k = line[2]
n= n+1
result = [-1]*(n+1)
for one in range(m):
    line = list(map(int,input().split()))
    weight[one] = line[1]
    money[one] = line[0]
result[0] = 0
result[1] = min(money)
result[1] = min(k,result[1])
for i in range(2,n+1):
    pass

