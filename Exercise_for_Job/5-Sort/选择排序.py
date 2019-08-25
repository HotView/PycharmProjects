a= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
len = len(a)
for i in range(len):
    index = i
    for j in range(i+1,len):
        if a[index]>a[j]:
            index = j
    tmp = a[i]
    a[i] = a[index]
    a[index] = tmp
print(a)
