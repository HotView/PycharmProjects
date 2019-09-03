from collections import defaultdict

t = int(input())
n_all = []
data_all = []
for i in range(t):
    n_all.append(int(input()))
    data_all.append(list(map(int,input().split())))
for n,data in zip(n_all,data_all):
    hashmap = defaultdict(int)
    for x in data:
        hashmap[x]+=1
    maxx = []
    for i in hashmap:
        maxx.append(hashmap[i])
    maxc = max(maxx)
    if maxc>n//2:
        print("NO")
    else:
        print("YES")
