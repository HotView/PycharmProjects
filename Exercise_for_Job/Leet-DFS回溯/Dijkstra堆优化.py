import heapq
import math
from collections import defaultdict
def dijkstra(graph,s):
    pqueue = []
    parent = {s:None}
    distance[s] = 0
    heapq.heappush(pqueue, (0, s))
    while (len(pqueue)>0):
        pair = heapq.heappop(pqueue)
        vertex = pair[1]
        dist_v = pair[0]
        if st[vertex]:
            continue
        st[vertex] = True
        for node in graph[vertex].keys():
                if dist_v+graph[vertex][node]<distance[node]:
                    distance[node] = dist_v+graph[vertex][node]
                    heapq.heappush(pqueue,(distance[node],node))
    return distance
graph = defaultdict(dict)
n,m,d,s = list(map(int,input().split()))
cakes = list(map(int,input().split()))
st = [False]*(n+2)
distance = [math.inf]*(n+2)
for i in range(m):
    x,y,z = list(map(int,input().split()))
    graph[x][y] = z
dist = dijkstra(graph,1)
print(dist)