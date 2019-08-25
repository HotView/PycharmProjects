def find(x):
    if p[x]!=x:
        p[x] = find(p[x])
    return p[x]
n,m = list(map(int,input().split()))
#初始化并查集
p = [0]*(n+2)
for i in range(1,n+1):
    p[i] = i
edges = []
for i in range(m):
    x,y,z = list(map(int,input().split()))
    edges.append([x,y,z])
edges.sort(key=lambda x:x[-1])
res = 0
cnt = 0
for i in range(m):
    a = edges[i][0]
    b = edges[i][1]
    w = edges[i][2]
    ori_a = find(a)
    ori_b =find(b)
    if ori_a!=ori_b:
        p[ori_a] = ori_b
        res+=w
        cnt+=1
if cnt<n-1:
    print("impossible")
else:
    print(res)