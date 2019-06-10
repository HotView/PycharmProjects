# 递归思想调用函数进行运算！
data = list(map(int,[i for i in input().split()]))
a= data[0]
b =  data[1]
m = data[2]
def fast_mod(a,b,m):
    result = 1
    while b!= 0:
        if(b&1) == 1:
            result = (result*a)%m
        b>>=1#b除以2
        a = (a*a)%m
    return result
print(fast_mod(a,b,m))