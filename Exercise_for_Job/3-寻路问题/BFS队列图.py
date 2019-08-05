from collections import defaultdict

graph = defaultdict(list)
graph["A"] = ["B","C"]
graph["B"]= ["A","C","D"]
graph["C"] = ["A","B","D","E"]
graph["D"] = ["B","C","F","E"]
graph["E"] = ["C","D"]
graph["F"] = ["D"]
print(graph)
def BFS(graph,s):
    queue = []
    visit = set()
    queue.append(s)
    visit.add(s)
    while (len(queue)>0):
        vertex = queue.pop(0)
        nodes = graph[vertex]
        for w in nodes:
            if w not in visit:
                queue.append(w)
                visit.add(w)
        print(vertex)
BFS(graph,"E")