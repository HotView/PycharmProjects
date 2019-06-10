#二维平面找出最大点
n = int(input())
array = []
for i in range(n):
    array.append([int(j) for j in input().split()])
max_p = []
limite = -1
array.sort(key=lambda x:x[0])
for i in range(n-1,-1,-1):
    if array[i][1]>limite:
        max_p.append(array[i])
        limite = array[i][1]
#print(max_p)
for k in reversed(max_p):
    print(k[0],k[1])

