from collections import  defaultdict
import heapq
N,M = map(int,input().split())
times = list(map(int,input().split()))
d = defaultdict(list)
for i in range(M):
    tmp = list(map(int,input().split()))
    for e in tmp[:-1]:
        d[tmp[-1]].append(e)
l_node = []
for i in range(N):
    heapq.heappush(l_node,[times[i],d[i+1],i+1])
print(l_node,"#######")
res = []
while len(l_node)>0:
    print("----------------------")
    print(l_node)
    temp = []
    node = None
    while len(l_node)>0:
        node = heapq.heappop(l_node)
        if len(node[1])>0:
            temp.append(node)
        else:
            res.append(node[-1])
            print(res,"res")
            break
    print(temp,"temp")
    for one in temp:
        heapq.heappush(l_node,one)
    print(node,"node")
    if node!=None:
        for i in range(len(l_node)):
            if node[-1] in l_node[i][1]:
                print(node[-1])
                l_node[i][1].remove(node[-1])
for x in res:
    print(x,end=' ')








