from collections import defaultdict
n,m = list(map(int,input().split()))
data = list(map(int,input().split()))
hashmap = defaultdict(int)
for x in data:
    hashmap[x]+=1
res = []
for x in data:
    if hashmap[x]>m:
        continue
    res.append(x)
for i in res:
    print(i)
