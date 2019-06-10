a= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
#不是严格的插入排序不是
n = len(a)
for i in range(1,n):
    tmp = a[i]
    j = i
    while j>0 and a[j-1]>tmp:
        a[j] = a[j-1]
        j =j-1
        print("paixu")
    a[j]=tmp#新牌落位
print(a)
