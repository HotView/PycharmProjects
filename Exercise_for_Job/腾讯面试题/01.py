MN= input()
mn = MN.split()
array = [int(i) for i in mn]
print(array)
n = int(mn[0])
k = int(mn[1])
count = 0
for _ in range(k):
    if n<=2:
        break
    n = (n//2)+1 if n & 1 else n//2
    count+=1
print(count+n)

