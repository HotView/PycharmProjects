n=100
p = [0]*n

def find(x):
    """
    :param x: 节点编号
    :return: 祖宗编号
    """
    if p[x]!=x:
        p[x] = find(p[x])
    return p[x]
## 初始化
for i in range(n):
    p[i] = i
## 将a集合合并到b集合
## p[find(a)] =find(b)
# 即a的祖宗等于b
a = 3
b = 5
for i in range(4):
    p[find(i)] =find(b)
for i in range(4):
    zozong = find(i)
    print(zozong)
