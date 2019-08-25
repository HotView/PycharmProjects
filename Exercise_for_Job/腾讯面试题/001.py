n,k = list(map(int,input().split()))
data = list(map(int,input().split()))
res=  0
#print(n,k)
#print(data)
min_v= sum(data[0:k])
temp_sum = min_v
for i in range(0,n-k):
    temp_sum = temp_sum-data[i]+data[i+k]
    #print("tempsum",temp_sum)
    if(temp_sum<min_v):
        min_v = temp_sum
        res = i+1
print(res+1)