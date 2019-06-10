import os
import docx
import sys
dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
i = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.docx' in file:
            if i <1:
                file_dir = roots+'/'+file
                docx_file = docx.Document(file_dir)
                print(file_dir)
                print(dir(docx_file))
                i = 1+1

