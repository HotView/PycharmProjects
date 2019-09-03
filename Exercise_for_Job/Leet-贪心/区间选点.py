# 先排序，按某个键值排序
# 试试
# 将每个区间按照右端点排序
# 从前往后枚举每个区间
#      如果当前区间已经包含点，则直接pass
#      否则选择当前区间的右端点
n = int(input())
lines = []
for i in range(n):
    lines.append(list(map(int,input().split())))
ans = []
lines.sort(key=lambda x:x[1])
end = -2e9
for x in lines:
    if x[0]>end:
        ans.append(x[1])
        end = x[1]
print(len(ans))