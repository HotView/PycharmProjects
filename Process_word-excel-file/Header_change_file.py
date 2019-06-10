import os
import docx
import sys
from header_chang import Header_change
OldStr = 'Q/YE/M'
NewStr= 'Q/HH/M'
dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
i = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.docx' in file:
            file_dir = roots+'/'+file
            file_dir = file_dir.replace('/','\\')
            Header_change(file_dir,OldStr,NewStr)
            i = 1+1
print(i)
