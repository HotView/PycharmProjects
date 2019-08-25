n = 10010
h = []*n# 存储的链表的头指针
e = []*n
ne = []*n
i = h[0]
x = 50
# i 是指针，ne[i]里面存的是指针i指向的节点的下一个结点的地址
while(i!=-1):
    if(e[i]==x):
        print("true")
        break
    i  =ne[i]