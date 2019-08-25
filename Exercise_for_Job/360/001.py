n = 0
data = []
def read_data():
    global n
    global data
    n = int(input())
    for i in range(n):
        line = list(map(int,input().split()))
        data.append(line)
def solution():
    global n
    global data
    for i in range(n):
        data_ = data[i]
        #print(data_)
        temp = data_[1]
        arr_numlow = []
        while(temp):
            rest = temp%data_[0]
            arr_numlow.append(rest)
            temp = temp//data_[0]
        temp = data_[2]
        arr_numhigh = []
        while (temp):
            rest = temp % data_[0]
            arr_numhigh.append(rest)
            temp = temp // data_[0]
        if(len(arr_numlow)!=len(arr_numhigh)):
            if arr_numhigh[-1]==data_[0-1]:
                print(len(arr_numhigh))
            else:print(len(arr_numhigh)-1)
        else:
            end = len(arr_numlow)-1
            while(arr_numlow[end]==arr_numhigh[end]):
                end = end-1
            print(end)
read_data()
print(data)
solution()
# 2
# 7 8 100
# 5 10 15