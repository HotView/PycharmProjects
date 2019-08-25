def lowbit(x):
    return x&(-x)
b = 7
res =0
while(b):
    b = b-lowbit(b)
    res+=1
print(res)