#构建树
class Tree:
    def __init__(self,value,left = 0,right = 0):
        self.value = value
        self.right = right
        self.left = left
N = int(input())
array = []
rootLR = []
index = []
for i in range(N-1):
    xy = list(map(int,input().split()))
    if 1 in xy:
        index.append(i)
    array.append(xy)
index_sort = index.sort()
count1 = index[1]-index[0]
count2 = N-index[1]
print(max(count1,count2))