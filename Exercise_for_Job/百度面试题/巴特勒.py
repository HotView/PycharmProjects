A = input()
B = input()
Q = input()
Q_int = int(Q)
LR_list = []
for i in range(Q_int):
    lr = input()
    l,r = lr.split()
    l = int(l)
    r = int(r)
    LR_list.append([l,r])
count_list = []
for i in range(Q_int):
    l = LR_list[i][0]
    r = LR_list[i][1]
    #tmp_A = A[l-1:r]
    #print(tmp_A)
    count = A.count(B,l-1,r)
    count_list.append(count)
for j in count_list:
    print(j)
