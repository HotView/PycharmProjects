line1 = [int(i) for i in input().split()]
line2 = [int(i) for i in input().split()]
do_list = []
for i in range(line1[1]):
    line3 = input().split()
    line3[1:] = [int(i) for i in line3[1:]]
    do_list.append(line3)
for do_one in do_list:
    if do_one[0]=="Q":
        max_score = 0
        if do_one[1]>do_one[2]:
            do_one[1],do_one[2] =do_one[2],do_one[1]
        for i in range(do_one[1]-1,do_one[2],1):
            max_score =max(max_score,line2[i])
        #print(sort_list)
        print(max_score)
    elif do_one[0]=="U":
        line2[do_one[1]-1] = do_one[2]
    else:
        continue

