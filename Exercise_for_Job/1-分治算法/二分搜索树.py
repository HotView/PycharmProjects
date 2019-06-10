## w
class Node:
    lft = None
    rgt = None
    def __init__(self,key,val):
        self.key =key
        self.val =val
def insert(node,key,val):
    if node is None:
        return Node(key,val)
    if node.key == key:
        node.val = val
    elif key< node.key:
        node.lft = insert(node.lft,key,val)
    else:
        node.rgt = insert(node.rgt,key,val)
    return node
def search(node,key):
    if node is None:
        raise KeyError
    if node.key==key:
        return node.val
    elif key < node.key:
        return search(node.lft,key)
    else:
        return search(node.rgt,key)

class Tree:
    root = None
    def __init__(self,key,val):
        self.root = insert(self.root,key.val)
    def __getitem__(self, key):
        return search(self.root.key)
    def __contains__(self, key):
        try :search(self.root,key)
        except KeyError:return False
        return True