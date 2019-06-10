def bfs():
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    inf = float("inf")
    dis = [[[[inf] * m for _ in range(n)] for _ in range(m)] for _ in range(n)]
    dis[startMe[0]][startMe[1]][startBox[0]][startBox[1]] = 0
    que = [[startMe[0], startMe[1], startBox[0], startBox[1]]]
    while que:
        p = que.pop(0)
        mx, my = p[0], p[1]
        bx, by = p[2], p[3]
        for i in range(4):
            newMx, newMy = mx + dx[i], my + dy[i]
            newBx, newBy = bx + dx[i], by + dy[i]
            if 0 <= newMx < n and 0 <= newMy < m and board[newMx][newMy] != 1:
                if newMx == bx and newMy == by:
                    if 0 <= newBx < n and 0 <= newBy < m and board[newBx][newBy] != 1 and dis[newMx][newMy][newBx][
                        newBy] == inf:
                        dis[newMx][newMy][newBx][newBy] = dis[mx][my][bx][by] + 1
                        if newBx == destination[0] and newBy == destination[1]:
                            return dis[newMx][newMy][newBx][newBy]
                        que.append([newMx, newMy, newBx, newBy])
                    else:
                        continue
                elif dis[newMx][newMy][bx][by] == inf:
                    dis[newMx][newMy][bx][by] = dis[mx][my][bx][by] + 1
                    que.append([newMx, newMy, bx, by])
                else:
                    continue
    return -1


def main():
    global board, n, m, startMe, startBox, destination
    s = input().strip().split()
    n, m = int(s[0]), int(s[1])
    board = [[0] * m for _ in range(n)]
    startMe = []
    startBox = []
    destination = []
    for i in range(n):
        s = input()
        for j in range(m):
            if s[j] == "#":
                board[i][j] = 1
            if s[j] == "E":
                destination = [i, j]
            if s[j] == "S":
                startMe = [i, j]
            if s[j] == "0":
                startBox = [i, j]
    print(bfs())


if __name__ == '__main__':
    main()