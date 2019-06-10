import re

#a = input()
b =input()
#c = input()

b_flags= re.findall('R\d+C\d+',b)
#c_flags= re.findall('R\d+C\d+',c)
if b_flags:
    b_re = re.split('(R|C)',b)
    row = int(b_re[2])
    col = int(b_re[4])
    row_chan =[]
    col_chan =[]
    while True:
        shang = row//26
        yu = row%26
        print("yu:",yu)
        row_chan.append(yu)
        if shang == 0:
            break
        row = shang
    #row_26 = row%26
    row_chan.reverse()
    print(row_chan)
    print(row,col)
    print('##############')
else:
    print(b)

