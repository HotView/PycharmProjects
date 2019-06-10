NK = input()
number = input()
nk = NK.split()
k = int(nk[1])
numbers = number.split()
num_list = []
num_no0_list = []
for i in numbers:
    num_list.append(int(i))
#num_list = np.array(num_list)
while(k):
    num_list.sort()
    for i,j in enumerate(num_list):
        if j ==0:
            num_list.pop(i)
        else:
            break
    print(num_list[0])
    for i,elemnet in enumerate(num_list):
        num_list[i] = num_list[i]-j
    k = k-1


