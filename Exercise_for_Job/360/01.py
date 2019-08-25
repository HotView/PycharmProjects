import random
import array
##2 2
##2 1
##1 1
## 20
n,m = map(int,input().split())
data = []
for i in range(n):
    data.append(list(map(int,input().split())))
#从四个面来考虑问题
max_hig =2
cemianji = 0
for h in range(1,max_hig+1):
    sx = 0
    sy = 0
    for i in range(n):
        for j in range(m):
            if data[i][j]>=h:
                sx = sx + 1
                break
    for j in range(m):
        for i in range(n):
            if data[j][i]>=h:
                sy = sy + 1
                break
    cemianji +=2*(sx+sy)
num_di = 0
for h in range(1,max_hig):
    for i in range(n):
        for j in range(m):
            if data[i][j]>0:
                num_di = num_di+1
dimianji = num_di*2
print(dimianji+cemianji)
