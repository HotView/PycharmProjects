import os
import docx
import sys
from win32com import client as wc
debug = False
def doSaveAas(dir_):
    word  = wc.Dispatch('Word.Application')
    doc = word.Documents.Open(dir_)
    if debug:
        print('hahahaahha')
    doc.SaveAs(dir_+'x', 12, False, "", True, "", False, False, False,False)
    doc.Close()
    word.Quit()
dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
i = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.doc' in file:
            file_dir_01 = (roots+'\\'+file).replace('/','\\')
            if debug:
                print(roots)
            doSaveAas(file_dir_01)
            if debug:
                print("**********")
            i = 1+1
    print(i)



