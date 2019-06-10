#邻接矩阵
def addEdge(a,u,v):
    a[v].append(u)
def print_graph(a):
    for i in range(len(a)):
        vector = []
        for j in range(len(a[i])):
            vector.append(a[i][j])
        print(vector)
"#####################################"
"存储列表必须在外部，访问周边的节点，然后递归。"
def DFS(a,node,visit):
    if a[node]:
        print(node)
        for j in a[node]:
            if j not in visit:
                visit.append(j)
                print(visit)
                DFS(a,j,visit)
"###################################"
def DFS_stack(a,node):
    stack = []
    visited = []
    stack.append(node)
    while(stack):
        tmp = stack.pop()
        visited.append(tmp)
        for ele in a[tmp]:
            if ele not in visited:
                stack.append(ele)

    print(visited)
"#######################################"
def BFS(a,node):
    queue = []
    visit = []
    queue.append(node)
    count = 1
    while(queue):
        tmp = queue.pop(0)
        visit.append(node)
        for son in a[tmp]:
            if son in visit:
                pass
            else:
                queue.append(son)
                count = count + 1
    return count
"########################################################"
if __name__ == '__main__':
    n= int(input())
    a= [[] for i in range(n)]

    for i in range(n-1):
        u,v = tuple(map(int,input().split()))
        addEdge(a,u-1,v-1)
    ans = -1
    for root in a[0]:
        cnt = BFS(a,root)
        print(cnt,"cnt")
        ans = max(ans,cnt)
    print(ans)
    print("########################DFS")
    ans = -1
    for root in a[0]:
        visit = []
        visit.append(root)
        DFS(a,root,visit)
        print(visit,"visit")
        ans = max(ans,len(visit))
    print(ans)
    print("###################DFS_stack")
    for root in a[0]:
        DFS_stack(a,root)