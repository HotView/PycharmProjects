# 寻找最短权值（快）的路径,最近的节点添加之后就不能再添加了
# 因为目前离最近顶点，里面的边都是正数
# 肯定不能通过第三个顶点中转，使得这个距离进一步缩短
# 需要三个散列表
# 不断更新散列表cost和parents
# 需要一个散列表将节点的所有邻居都存储在散列表中,散列表包含散列表
from collections import defaultdict
graphy= defaultdict(dict)
#graphy["start"] = {}
all_nodes = set()
N = int(input())
for i in range(N):
    a,b,c = input().split()
    c = int(c)
    graphy[a][b] = c
    all_nodes.add(a)
    all_nodes.add(b)
#graphy[]
print(all_nodes)
print(graphy.keys())

infinity = float("inf")
costs = {}
for des in all_nodes:
    costs[des] = infinity
for neigh in graphy["start"].keys():
    costs[neigh] = graphy["start"][neigh]
print(costs)
# parents = {}
# parents["a"] = "start"
# parents["b"] = "start"
# parents["fin"] = None
processed = []

def find_lowest_node(costs):
    #在未处理的节点中找出开销最小的节点
    lowest_cost = float("inf")
    lowest_cost_node = None
    for node in costs:
        cost = costs[node]
        if cost<lowest_cost and node not in processed:
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node
nearest_node = find_lowest_node(costs)
while nearest_node is not None:
    cost = costs[nearest_node]
    neifhbors = graphy[nearest_node]
    for n in neifhbors.keys():
        new_cost = cost+neifhbors[n]
        if costs[n]>new_cost:
            costs[n] = new_cost
            #parents[n] = nearest_node
    processed.append(nearest_node)
    nearest_node = find_lowest_node(costs)
print(costs)
print(processed)






