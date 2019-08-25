def read_data():
    global k
    global n
    for i in range(c):
        _n,_k = list(map(int,input().split()))
        n.append(_n)
        k.append(_k)

def resolution():
    global k
    global n
    for i in range(c):
        temp_n = n[i]
        temp_k =k[i]
        nums = []
        for i in range(temp_n):
            nums.append(i+1)
        remain = temp_n
        count =1
        while remain>2:
            for i in range(temp_n):
                if nums[i]!=0:
                    if count==temp_k:
                        count = 1
                        continue
                    elif count==1:
                        nums[i] = 0
                        remain-=1
                    count += 1


        for x in nums:
            if x!=0:
                print(x,end=" ")
        print()
c = int(input())
n = []
k = []
read_data()
resolution()
