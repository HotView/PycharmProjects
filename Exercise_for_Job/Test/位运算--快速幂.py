import heapq
data = list(map(int,input().split()))
a = data[0]
b = data[1]
p = data[2]
res = 1%p
while(b):
    if b&1:
        res = res**a%p
    a= a*a%p
    b>>=1#去除个位
print(res)