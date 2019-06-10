import numpy as np

n= input()
m = input()
k = input()
n = int(n)
m = int(m)
k= int(k)

a = np.zeros((n,m))
list = []
for i in range(n):
    for j in range(m):
        a[i,j]=int((i+1)*(j+1))
        list.append(a[i,j])

list.sort()
print(list[k-1])