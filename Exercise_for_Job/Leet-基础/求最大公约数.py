def gcd(a,b):
    if b:
        return gcd(b,a%b)
    else:
        return a
print(gcd(24,60))