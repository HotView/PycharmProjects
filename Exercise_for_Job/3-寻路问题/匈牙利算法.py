h = [-1]*20#头结点指针，默认值为-1
idx = 0#当前指向的节点，默认值为0，新建节点的地址
e= [0]*20 #节点的数值，索引是指针
ne = [-1]*20# 指针的下一个指针
head =-1#存放的是下一节点的索引
def insert(a):
    global head
    global idx
    e[idx] =a
    ne[idx] = head
    head = idx
    idx+=1
for i in range(0,10):
    insert(i)
pr = head
while(pr!=-1):
    ver = e[pr]
    print(ver)
    pr = ne[pr]
