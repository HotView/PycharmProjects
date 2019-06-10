"""4
1 1
2 2
3 3
1 3
4
1 1 2 2
1 1 3 3
2 2 3 3
1 2 2 3
2
4
2
2"""
n= int(input())
points = []
for i in range(n):
    point = list(map(int,input().split()))
    points.append(point)
#print(points)
points_sort = sorted(points,key=lambda x:x[0])
#print(points_sort)
m = int(input())

rects = []
results = []
for count in range(m):
    res = 0
    rect = list(map(int,input().split()))
    for i in range(n):
        if points_sort[i][0]<rect[0]:
            i = i+1
        else:
            break
    for j in range(i,n):
         if points_sort[j][0]<=rect[2]:
             j=j+1
         else:
            break
    for k in range(i,j):
        if points_sort[k][1]>=rect[1] and points_sort[k][1]<=rect[3]:
            res = res+1
    results.append(res)
for one in results:
    print(one)
