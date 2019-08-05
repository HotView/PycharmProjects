## 需要一个优先队列，需要一个距离字典，需要一个状态集合，父节点的列表。
import heapq
import math
graph = {
    "A":{"B":5,"C":1},
    "B":{"A":5,"C":2,"D":1},
    "C":{"A":1,"B":2,"D":4,"E":8},
    "D":{"B":1,"C":4,"E":3,"F":6},
    "E":{"C":8,"D":3},
    "F":{"D":6}
}
def init_distance(graph,s):
    distance = {s:0}
    for vertex in graph:
        if vertex!=s:
            distance[vertex] = math.inf
    return distance
def dijkstra(graph,s):
    pqueue = []
    heapq.heappush(pqueue,(0,s))
    seen =set()
    parent = {s:None}
    distance = init_distance(graph, s)
    while (len(pqueue)>0):
        pair = heapq.heappop(pqueue)
        vertex = pair[1]
        dist = pair[0]
        seen.add(s)
        for node in graph[vertex].keys():
            if node not in seen:
                if dist+graph[vertex][node]<distance[node]:
                    distance[node] = distance[vertex]+graph[vertex][node]
                    heapq.heappush(pqueue,(distance[node],node))
                    parent[node] = vertex
    return distance,parent
parent,distance = dijkstra(graph,"A")
print(parent)
print(distance)


