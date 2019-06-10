n = input()
array = list(map(int,input().split()))
array.sort()
right = 0
index = 0
res = 0
leng = len(array)
print(array)
for num in array:
    res = max(res,num*sum(array[index:]))
    index = index+1
print(res)
