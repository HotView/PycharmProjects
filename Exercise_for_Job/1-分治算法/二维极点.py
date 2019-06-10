"""8
2 4
3 10
5 3
6 8
8 2
10 6
13 5
15 7"""
def two_dim_maxium(arr):
    if(len(arr)==1):
        return arr
    l_arr,r_arr = part_arr(arr)
    print("l_arr",l_arr)
    print("r_arr", r_arr)
    L_max = two_dim_maxium(l_arr)
    R_max = two_dim_maxium(r_arr)
    c_point = R_max[0]
    L_max = update(L_max,c_point)
    maxPoints =L_max+R_max
    print("merge",maxPoints)
    return maxPoints
def part_arr(arr):
    left = 0
    right = len(arr)
    mid = (left+right)>>1
    l_arr = arr[left:mid]
    r_arr = arr[mid:]
    return l_arr,r_arr
def update(arr,point):
    new_arr = []
    for one in arr:
        if one[1]>=point[1]:
            new_arr.append(one)
    return new_arr
N = int(input())
points = []
for i in range(N):
    line = list(map(int,input().split()))
    points.append(line)
points = sorted(points,key=lambda x:x[0])
print(points)
res = two_dim_maxium(points)
print(res)

