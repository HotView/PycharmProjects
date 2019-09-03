n = int(input())
data = []
for i in range(n):
    w,s = list(map(int,input().split()))
    data.append([w+s,w,s])
data.sort()
res = -2e9
sum = 0
for i in range(n):
    res=max(res,sum-data[i][2])
    sum +=data[i][1]
print(res)