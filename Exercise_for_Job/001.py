from collections import defaultdict
hashmap = defaultdict
n = int(input())
exp = input()
data = exp.split()
datanum = [data[i] for i in range(0,len(data),2)]
dataop = [data[i] for i in range(1,len(data),2)]
length = len(data)
for i in range(100):
    j = 1
    while(j<length):
        if data[j]=="*":
            if int(data[j-1])>int(data[j+1]):
                data[j+1],data[j-1] = data[j-1],data[j+1]
        j=j+2
    k = 1
    while(k+2<length):
        if k<2:
            if data[k] == "+" and (data[k+2]!="+" or data[k+2]!="-"):
                if int(data[k-1])>int(data[k+1]):
                    data[k + 1], data[k - 1] = data[k - 1], data[k + 1]
        else:
            if data[k-2]=="+" and data[k] == "+" and (data[k+2]=="+" or data[k+2]=="-"):
                if int(data[k-1])>int(data[k+1]):
                    data[k + 1], data[k - 1] = data[k - 1], data[k + 1]
        k=k+2
res = ""
for x in data:
    res+=x+" "
print(res)


