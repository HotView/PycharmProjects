# 将所有区间按左端点从小到大排序
# 从前往后依次枚举每个区间，在所有能覆盖的start的区间中，选择右端点最大的区间
# 然后更新start的数值
st,end = list(map(int,input().split()))
n = int(input())
lines = []
for i in range(n):
    lines.append(list(map(int,input().split())))
lines.sort()
res = 0
i  =0
success = False
while(i<n):
    j  = i
    r = -2e9
    # 遍历所有左端点在start的左边，右端点的最大值
    while(j<n and lines[j][0]<=st):
        r = max(r,lines[j][1])
        j +=1
    if(r<st):
        res = -1
        break
    res+=1
    if r>=end:
        success = True
        break
    st = r
    i = j-1
if not success:
    res =-1
print(res)
