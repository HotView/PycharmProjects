#单调队列、栈
def resolution():
    line = data[0]
    lengths=  len(line)
    max_val = max(line)
    min_val = min(line)
    res= 0
    i = 0
    while i<lengths:
        if i+1<lengths and line[i]-line[i+1]>0:
            res = res+line[i]-line[i+1]
            i =i+1
        else:
            i = i+1
    print(100*res)
n,k,s = list(map(int,input().split()))
data = []
for i in range(s):
    line =list(map(int,input().split()))
    data.append(line)
resolution()