## 需要一个优先队列，需要一个距离字典，需要一个状态集合，父节点的列表。
import heapq
import math
from collections import defaultdict
graph = defaultdict(dict)
graph = {
    "A":{"B":5,"C":1},
    "B":{"A":5,"C":2,"D":1},
    "C":{"A":1,"B":2,"D":4,"E":8},
    "D":{"B":1,"C":4,"E":3,"F":6},
    "E":{"C":8,"D":3},
    "F":{"D":6}
}
def init_distance(graph,s):
    for vertex in graph:
        distance[vertex] = math.inf
    distance[s] = 0
    return distance
def dijkstra(graph,s):
    pqueue = []
    parent = {s:None}
    seen = set()
    distance = init_distance(graph, s)
    heapq.heappush(pqueue, (0, s))
    while (len(pqueue)>0):
        pair = heapq.heappop(pqueue)
        vertex = pair[1]
        dist_v = pair[0]
        if vertex  in seen:
            continue
        seen.add(vertex)
        for node in graph[vertex].keys():
                if dist_v+graph[vertex][node]<distance[node]:
                    distance[node] = dist_v+graph[vertex][node]
                    heapq.heappush(pqueue,(distance[node],node))
                    parent[node] = vertex
    return distance,parent
parent,distance = dijkstra(graph,"A")
print(parent)
print(distance)
n,m = list(map(int,input().split()))
all_nodes = set()
for i in range(m):
    x,y,z = list(map(int,input().split()))
    all_nodes.add(x)
    all_nodes.add(y)
    graph[x][y] = z

