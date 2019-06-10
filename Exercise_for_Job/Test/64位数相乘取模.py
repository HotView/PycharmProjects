"""
a*b
a+a+a+a+a+a...+a
a*1 = a
a*2 = 2a
a*4 = 4a
a*8= 8a
a*($2^k$) = $2^k$a
将b换为二进制数
"""
data = list(map(int,input().split()))
a = data[0]
b = data[1]
p = data[2]
res = 0
while(b):
    if b&1:
        res = (res+a)%p
    b>>=1
    a= a*2%p
print(res)

