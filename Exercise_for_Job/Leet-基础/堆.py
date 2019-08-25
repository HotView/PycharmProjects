def down(u):
    t = u
    if(u*2<=size and h[u*2]<h[t]):
        t = u*2
    if (u * 2 <= size and h[u * 2+1] < h[t]):
        t = u * 2+1
    if(u!=t):
        h[u],h[t] = h[t],h[u]
        down(t)
def up(u):
    while(u//2  and h[u//2]>h[u]):
        h[u//2],h[u] = h[u],h[u//2]
        u//=2
n,m = list(max(int,input().split()))
h = list(max(int,input().split()))
size = len(h)
i = n//2
while(i):
    down(i)
    i -=1
while(m):
    h[i]=h[size]
    size-=1
    down(1)
    m-=1
