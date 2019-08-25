N= 1010
son = [[]*26 for i in range(N)]
cnt = []*N
idx = 0
string = input()
base = ord('a')
int
p = 0
for i in range(len(string)):
    u = ord(string[i])-base
    if not son[p][u]:
        idx+=1
        son[p][u] =idx
        p =son[p][u]