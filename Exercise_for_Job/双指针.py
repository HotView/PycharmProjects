import sys
data  = []
n = int(input())
for line in range(n):
    a=sys.stdin.readline()
    data.append(a)
for a in data:
    slow = 0
    a= list(a)
    for i in range(len(a)):
        a[slow] = a[i]
        slow = slow+1
        if slow>=3 and a[slow-3]==a[slow-2] and a[slow-2]==a[slow-1]:
            slow = slow-1
        elif slow>=4 and a[slow-4]==a[slow-3] and a[slow-2]==a[slow-1]:
            slow =slow-1
    print(''.join(a[:slow-1]))

