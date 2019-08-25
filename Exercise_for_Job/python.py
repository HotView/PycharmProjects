e = [0]*20
ne = [-1]*20
idx = 0
head = -1
def insert(x):
    global head
    global idx
    e[idx] =x
    ne[idx] = head
    head = idx
    idx+=1
for i in range(10):
    insert(i)
pr = head
while(pr!=-1):
    j = e[pr]
    print(j)
    pr = ne[pr]
