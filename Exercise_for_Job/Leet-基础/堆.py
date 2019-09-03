def down(pr):
    pt = pr
    l = 2*pr
    if(l<=size and heap[l]<heap[pt]):
        pt = l
    r = 2*pr+1
    if (r <= size and heap[r]<heap[pt]):
        pt = r
    if(pr!=pt):
        heap[pt],heap[pr] = heap[pr],heap[pt]
        down(pt)
def up(pr):
    while(pr//2 and heap[pr//2]>heap[pr]):
        heap[pr//2],heap[pr] =heap[pr],heap[pr//2]
        pr = pr//2

n,m = list(map(int,input().split()))
heap = [0]*(n+1)
heap[1:] = list(map(int,input().split()))
size = n
for i in range(n//2,0,-1):
    down(i)
while(m):
    print(heap[1],end=" ")
    heap[1] = heap[size]
    size -= 1
    down(1)
    m-=1



