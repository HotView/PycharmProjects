class Heap():
    def __init__(self):
        self._heap = []
        self._size = 0
    def add(self,item):
        self._size+=1
        self._heap.append(item)#先添加进去，再进行调整
        curPos = len(self._heap)-1
        while curPos>0:
            parent = (curPos-1)//2 #Integer quotient
            parentItem = self._heap[parent]
            if parentItem<=item:
                break
            else:
                self._heap[curPos]=  self._heap[parent]
                self._heap[parent] =item
                curPos = parent
    def pop(self):
        self._size-=1
        topItem = self._heap[0]
        bottomItem = self.pop(len(self._heap)-1)
        if len(self._heap)==0:
            return bottomItem

        self._heap[0] = bottomItem
        lastIndex = len(self._heap)-1
        curPos = 0
        while True:
            leftChild = 2*curPos+1
            rightChild = 2 * curPos + 2
            if leftChild>lastIndex:
                break
            if rightChild>lastIndex:
                maxchild = leftChild
            else:
                leftItem = self._heap[leftChild]
                rightItem = self._heap[rightChild]
                if leftItem<rightItem:
                    maxchild = leftChild
                else:
                    maxchild = rightChild
            maxItem = self._heap[maxchild]
            if bottomItem<=maxItem:
                break
            else:
                self._heap[curPos] = self._heap[maxchild]
                curPos = maxchild
        return topItem



