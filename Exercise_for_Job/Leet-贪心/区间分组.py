# 左端点排序
# 枚举区间，判断每个区间能否放到已有组中
# 如果不存在，新建一个组
# 如果存在，将其放进去，更新max_r
# 动态维护最小值
import heapq
n = int(input())
lines = []
for i in range(n):
    lines.append(list(map(int,input().split())))
lines.sort()
max_r = []
heapq.heapify(max_r)
for x in lines:
    if not max_r or max_r[0]>=x[0]:
        heapq.heappush(max_r,x[1])
    else:
        heapq.heappop(max_r)
        heapq.heappush(max_r,x[1])
print(len(max_r))
