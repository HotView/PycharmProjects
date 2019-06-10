def insertHash(hash,m,data):#m是哈希表的长度
    i = data%13#计算哈希地址
    while(hash[i]):#元素位已被占用
        i = i+1%m  #线形探测法解决冲突
    hash[i] = data
def crreatHash(hash,m,data,n):
    for i in range(n):
        insertHash(hash,m,data)
def SearchHash(hash,m,key):
    i = key%13
    while(hash[i]and hash[i]!=key):#判断是否冲突
        i = (i+1)%m#线形探测法
    if(hash[i]==0):#查到开放单元，查找失败
        return -1
    else:
        return i;#查找成功，返回元素的下标索引。

