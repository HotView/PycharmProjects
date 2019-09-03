n = int(input())
data = list(map(int,input().split()))
data.sort()
res = 0
pres=  [0]
for i in range(n-1):
    res+=data[i]
    pres.append(res)
print(sum(pres))