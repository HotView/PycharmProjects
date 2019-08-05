from collections import defaultdict

graph = defaultdict(list)
graph["A"] = ["B","C"]
graph["B"]= ["A","C","D"]
graph["C"] = ["A","B","D","E"]
graph["D"] = ["B","C","E","F"]
graph["E"] = ["C","D"]
graph["F"] = ["D"]
def DFS(graph,s):
    stack = []
    visit = set()
    stack.append(s)
    visit.add(s)
    while(len(stack)>0):
        vertex = stack.pop()
        nodes = graph[vertex]
        for x in nodes:
            if x not in visit:
                stack.append(x)
                visit.add(x)
        print(vertex)
DFS(graph,"A")