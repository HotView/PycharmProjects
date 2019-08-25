s = input()
i = 0
while(i<len(s)):
    j = i
    while(j<len(s) and s[j]!=" "):
        j = j+1
    #j指向空格的位置
    print(s[i:j])
    while(j<len(s) and s[j]==" "):
        j +=1
    # j指向非空格的位置
    i = j
