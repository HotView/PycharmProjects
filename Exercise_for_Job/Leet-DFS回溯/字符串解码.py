def dfs(s):
    res = ""
    if not len(s):
        return res
    i = 0
    while (i < len(s)):
        j = i
        while j < len(s) and s[j].isalpha():
            res += s[j]
            j += 1
        t = 0
        while (j < len(s) and s[j].isdigit()):
            t = t * 10 + int(s[j])
            j += 1
        if j<len(s) and s[j] == '[':
            sum = 1
            k = j +1
            while (sum > 0):
                if s[k] == '[':
                    k += 1
                    sum += 1
                elif s[k] == "]":
                    k += 1
                    sum -= 1
                else:
                    k += 1
            print(s[j + 1:k - 1],"-")
            res += dfs(s[j + 1:k - 1]) * t
            j = k
        i = j
    return res
s = input()
res = dfs(s)
print(res)