s = "1"
n = int(input())-1
i = 0
res =["1"]
for i in range(n):
    i = 0
    temp = ""
    while(i<len(s)):
        k = i
        while(k<len(s) and s[k] == s[i]):
            k =k+1
        temp+=str((k-i))+s[i]
        i = k
    s = temp
    res.append(temp)
print(res)
