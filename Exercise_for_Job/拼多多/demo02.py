import sys
def solution():
    a = sys.stdin.readline().split()
    for i in range(len(a)):
        for j in range(i):
            a[i],a[j] = a[j],a[i]
            b = a[1:]
            c = b.append(a[0])
            lens = len(a)
            flag = 1
            for i in range(lens):
                if a[i][-1]!=b[i][0]:
                    flag = 0
                    break
            if flag:
                print("true")
                return 0
    print("false")
solution()