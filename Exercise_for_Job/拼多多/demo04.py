import sys
# 10
# 1 2 3 4 5 6 7 8 9 10
# 1 1 1 1 1 1 1 1 1 10
# 动规思路，先按边长排序，然后从最大的边开始计算，要不要添加第块板子，添加，不添加，dp算法。
class box():
    def __init__(self,length,weight):
        self.length = length
        self.weight = weight

def solution():
    boxes= []
    N = int(sys.stdin.readline())
    Li = list(map(int, sys.stdin.readline().split()))
    Wi = list(map(int, sys.stdin.readline().split()))
    for i,j in zip(Li, Wi):
        boxes.append(box(i,j))
    boxes = sorted(boxes,key=lambda x:x.length)
    for x in boxes:
        print(x.length,x.weight)
    max_res= 1
    for i in range(1,len(boxes)):
        pass

solution()
