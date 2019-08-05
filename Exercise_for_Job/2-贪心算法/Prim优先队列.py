# 一个存储所有边（分方向）的字典索引结构，一个优先队列，一个集合，一个结果列表。
from heapq import heapify,heappop,heappush
from collections import defaultdict
import math
def prim(edges):
    conn = defaultdict(list)
    ## 初始化，将所有的边分方向初始化，即每个字典元素存储的就是以这个点为起始的所有边。
    start = None
    start_edge =math.inf
    for n1, n2, c in edges:
        if c<start_edge:
            start = n1
            start_edge = c
        conn[n1].append((c, n1, n2))
        conn[n2].append((c, n2, n1))
    print(start)
    mst = []
    used = set(start)
    usable_edges = conn[start][:]
    heapify(usable_edges)

    while usable_edges:
        cost,n1,n2 = heappop(usable_edges)
        if n2 not in used:
            used.add(n2)
            mst.append((n1,n2,cost))
            for e in conn[n2]:
                if e[2] not in used:# 关键精髓的点
                    heappush(usable_edges,e)
    return mst
nodes = list("ABCDEFG")
edges = [ ("A", "B", 7), ("A", "D", 5),
               ("B", "C", 8), ("B", "D", 9),
               ("B", "E", 7), ("C", "E", 5),
               ("D", "E", 15), ("D", "F", 6),
               ("E", "F", 8), ("E", "G", 9),
               ("F", "G", 11)]
print ("prim:", prim( edges))
