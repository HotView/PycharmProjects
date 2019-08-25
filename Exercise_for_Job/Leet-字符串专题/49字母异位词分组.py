from collections import defaultdict
hashmap = defaultdict(list)
data = input().split()
for x in data:
    sort_x = ''.join(sorted(x))
    hashmap[sort_x].append(x)
res = []
for x in hashmap.keys():
    res.append(hashmap[x])
print(res)
