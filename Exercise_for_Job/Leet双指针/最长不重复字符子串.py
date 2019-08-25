from collections import defaultdict
hashmap = defaultdict(int)
n = int(input())
s= list(map(int,input().split()))
i = 0
j = 0
res = 0
for i in range(n):
    hashmap[s[i]]+=1
    while(hashmap[s[i]]>1):
            hashmap[s[j]]-=1
            j = j+1
    res = max(res,i-j+1)
print(res)
