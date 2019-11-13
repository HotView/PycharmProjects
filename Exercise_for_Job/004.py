def solution():
    data = list(map(int,(input().split())))
    n = len(data)
    l  =0
    r = n-1
    while(l<r):
        if data[l]!=data[r]:
            print(False)
            return
        l = l+1
        r = r-1
    print(True)
solution()
