# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
myquene = []
class Solution:
    def levelOrder(self, root):
        res = []
        self.pre(root,0,res)
        print(res)
        return res
    def pre(self,root,depth,res):
        if not root:
            return
        if depth>len(res):
            res.append([])
        res[depth].append(root.val)
        self.pre(root.left,depth+1,res)
        self.pre(root.left, depth + 1, res)
