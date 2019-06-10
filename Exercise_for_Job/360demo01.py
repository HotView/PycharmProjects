N  = int(input())
data = list(map(int,input().split()))
b= set(data)
number = dict()
if N==1:
    print(1)
if N<1:
    print(0)
for i in b:
    number[i] = data.count(i)
count = list(number.values())
count.sort()
if count[0]==count[-1]:
    print(len(b))
elif count[-1]%count[0]==0:
    print(count[-1]/count[0]+len(b)-1)
else:
    print(0)
