n = int(input())
data = list(map(int,input().split()))
if n==3:
    print(1)
    exit()
else:
    data_index = []
    for index,j in enumerate(data):
        if j==0:
            data_index.append(index)
count = 0
for i in data_index:
    if data[i+1]!=0 and data[i-1]!=0:
        count = count+max(data[i+1],data[i-1])
    if data[i+1]==0 and data[i-1]!=0:
print(count)

