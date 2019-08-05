import sys
def solution():
    a = list(map(int,sys.stdin.readline().split()))
    b = list(map(int,sys.stdin.readline().split()))
    if not a:
        print('NO')
        return
    long_ = len(a)
    for i in range(long_-1):
        if a[i]>=a[i+1]:
            break
    index = i+1
    pre = a[i]
    next = a[i+2]
    c =sorted(b,reverse=True)
    #print(c)

    for x in c:
        if pre<x<next:
            a[index]=x
            res= ''
            # for i in a:
                # print(type(i))
                # print(str(i).join())
            print(''.join(a))
            return 0
    print("NO")
solution()