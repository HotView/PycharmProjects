from array import array

class arrayStack():
    def __init__(self):
        self._items = array()
    def __iter__(self):
        cursor = 0
        while cursor <len(self):
            yield self._items[cursor]
            cursor+=1
    def peek(self):
        return self._items[len(self)-1]
    def clear(self):
        self._size = 0
        self._items = array()
    def push(self,item):
        self._items[len(self)] = item
        self._size+=1
    def pop(self):
        olditem = self._items[len(self)-1]
        self._size-=1