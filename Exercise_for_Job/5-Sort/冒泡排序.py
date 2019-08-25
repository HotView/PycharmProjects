a= [8,5,5,5,5,54,1,2,21,0,21,13,13,2]
len = len(a)
for i in range(5,1,-1):
    print(i)
#左闭右开的集合来进行调试
for i in range(len):
    for j in range(len-i-1):
        if a[j]>a[j+1]:
            tmp = a[j]
            a[j] = a[j+1]
            a[j+1] = tmp
print(a)
