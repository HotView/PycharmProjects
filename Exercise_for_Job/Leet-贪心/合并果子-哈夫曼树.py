# 首先合并最小的两堆
# 依次循环上述的这个步骤
import heapq
n = int(input())
fruit = list(map(int,input().split()))
heapq.heapify(fruit)
res = 0
while(len(fruit)>1):
    t1 = heapq.heappop(fruit)
    t2 = heapq.heappop(fruit)
    t3 = t1+t2
    res = res + t3
    heapq.heappush(fruit,t3)
print(res)
