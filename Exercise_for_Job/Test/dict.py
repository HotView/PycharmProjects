#字典实现父节点的追踪
a={}
class Point:
    def __init__(self,x,y):
        self.x  =x
        self.y = y
b = Point(1,2)
c= 5
a[b] = 1
a[c] = 6
print(a[c])