# 将数据分为两拨：最大值之内的和最大值之外的
# 最大值之内的就不计算距离
# 最大值之外的就计算距离
# 比较之内的和之外的那个大，然后最大的那个即可。
import heapq

heap = []
n = int(input())
d= [0]*(n+1)
v= [0]*(n+1)
d[1:] =list(map(int,input().split()))
v[1:] = list(map(int,input().split()))
heapq.heappush(heap,(0,0))
ans=0
now = 0
for i in range(n):
    next = now
    maxx = heap[0][1]
    for j in range(now+1,n+1):
        if v[j]+(d[j]-d[now])*2>maxx:
            maxx = v[j]+(d[j]-d[now])*2
            next = j
    if now!=next:
        heapq.heappush(heap,(-maxx,maxx))
    for j in range(now + 1, next):
        heapq.heappush(heap,(-v[j], v[j]))
    now = next
    _,s_top = heapq.heappop(heap)
    ans+=s_top
    print(ans)