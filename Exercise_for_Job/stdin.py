#对于不确定输入的行数的，可以先用stdin读取所有，然后一行一行的读取数据
import sys
for i in sys.stdin:
    print(i)

if __name__ == "__main__":
    strList = []
    for line in sys.stdin:  #当没有接受到输入结束信号就一直遍历每一行
        tempStr = line.split()#对字符串利用空字符进行切片
        strList.extend(tempStr)#把每行的字符串合成到列表
"针对多行输入的数据，不确定输入的行数，可以使用下述程序。"
import sys
data_list = []
for line in sys.stdin:
    data  = list(map(int,line.split()))
    data_list.append(data)

for line in data_list:
    print(line[0]+line[1])
