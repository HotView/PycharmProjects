n = int(input())
T = input()
m = int(input())
data = []
for i in range(m):
    data.append(input())
lens = len(T)
count = 0
for x in data:
    temp = x+x
    len2 = len(temp)
    if len2>lens:
        if temp[:lens]==T:
            count+=1
    else:
        if temp==T[:len2]:
            count+=1
print(count)