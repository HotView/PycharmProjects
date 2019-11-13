from collections import defaultdict
def solution(s1,s2):
    len1 = len(s1)
    len2 = len(s2)
    dp = [[0 for ii in range(len2+1)] for _ in range(len1+1)]
    for i in range(len1+1):
        for j in range(len2+1):
            if i==0:
                f[i][j] = j
                continue

hashmap = defaultdict(int)
n =int(input())
for i in range(1,n):
    for j in range(1,n):
        pre = i
        cur = j
        cnt = 2
        while(cur<=n):
            res = cur + pre
            pre = cur
            cur = res
            cnt+=1
            if cur==n:
                hashmap[cnt]+=1
data = []
for x in hashmap.keys():
    data.append([x,hashmap[x]])
data.sort()
for x in data:
    print(x[0],x[1])





