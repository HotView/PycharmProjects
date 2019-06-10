import heapq
heap = []
a= [2,5,53,6,6,6,526,23,2]
for i in a:
    heapq.heappush(heap,i)
print(heap)
b = heapq.heappop(heap)
print(b)
print(heap)