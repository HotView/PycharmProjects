#二维平面找出最大点
n = int(input())
array = []
for i in range(n):
    array.append([int(j) for j in input().split()])
max_p = []
limite = -1
array.sort(key=lambda x:x[1])
for i in range(n-1,-1,-1):
    if array[i][0]>limite:
        print(array[i][0],array[i][1])
        limite = array[i][0]
