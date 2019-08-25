n,m,x = list(map(int,input().split()))
s1 = list(map(int,input().split()))
s2= list(map(int,input().split()))
i = 0
j = m-1
for i in range(n):
    while(j>=0 and s1[i]+s2[j]>x):
        j-=1
    if s1[i]+s2[j]==x:
        print(i,j)