#每个点都只是遍历一次,用一个状态来进行剪枝
N= 100010
M = N*2
h = []*N
e = []*M
ne = []*M
st = []*N
# idx 指向节点的指针，也就是数据节点数组的索引，一直增加的
idx = 0
# d和q是bfs需要的数据
#距离
d = []*N
#节点
q =[]*N
def add(a,b):
    global idx
    e[idx] = b
    ne[idx] = h[a]
    h[a] = idx
    idx+=1
def dfs(u):
    st[u] = True
    i = h[u]
    while(i!=-1):
        j = e[i]
        if(not st[j]):
            dfs(j)
        i = ne[i]
def bfs():
    hh = 0
    tt = 0
    q[0] =1
    d[1] = 0
    while(hh<tt):
        t = q[hh]
        hh+=1
        i = h[t]
        while(i!=-1):
            j = e[i]
            if(d[j]==-1):
                d[j] = d[t]-1
                tt+=1
                q[tt] = j