n = int(input())
m = int(input())
def dp(n,m):
    print(n,m)
    if m==1:
        print("m==1")
        return 1
    elif m>n:
        print("m>n")
        return 0
    else:
        return dp(n-1,m-1)+dp(n-m,m)
count = dp(n,m)
print(count)

