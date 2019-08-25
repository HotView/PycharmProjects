N = 1000010
n = int(input())
segs = []
for i in range(n):
    segs.append(list(map(int,input().split())))
res = []
segs.sort()
st = -2e9
ed = -2e9
res = []
for seg in segs:
    if ed<seg[0]:
        #没有交集
        if(st!=-2e9):#过滤第一个元素
            res.append([st,ed])
        st = seg[0]
        ed = seg[1]
    else:
        # 有交集
        ed = max([ed,seg[1]])
if(st!=-2e9):
    res.append(st,ed)
print(len(res))