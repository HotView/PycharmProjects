import os
import docx
import sys
from win32com import client as wc
debug = False

dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
i = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.doc' in file:
            if '.docx' in file:
                pass
            else:
                file_dir_01 = (roots + '/' + file)
                os.remove(file_dir_01)
                i= i+1
                print(i)



