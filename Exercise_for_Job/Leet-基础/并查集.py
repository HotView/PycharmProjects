N = 10010
p=[0]*N
for i in range(1,N):
    p[i] = i
def find(x):
    #返回集合的编号
    if(p[x]!=x):
        p[x] = find(p[x])
    return p[x]
n,m = list(max(int,input().split()))
a,b = list(max(int,input().split()))
op = []
while(m):
    if(op[0]=='M'):
        #合并
        p[find(a)] = find(b)
    else:
        if(find(a)==find(b)):
            print("yes")
        else:
            print("no")
    m-=1
