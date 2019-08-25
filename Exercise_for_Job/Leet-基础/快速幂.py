def qmi(a,b):
    res =1
    base = a
    while(b):
        if(b&1):
            res = res*base
        base = base*base
        b = b>>1
    return res
print(qmi(3,4))