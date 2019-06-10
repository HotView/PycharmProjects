import os
import docx
import sys
from win32com import client as wc
dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
i = 0
OldStr ='地址：广东省佛山市南海区狮山镇松岗桃园东路19号B楼209室'
NewStr ='地址：湖北省鄂州市梁子湖区凤凰大道特一号'
def Text_chang(docx_dir,OldStr,NewStr):
    w = wc.Dispatch('Word.Application')
    doc = w.Documents.Open(file_dir)
    w.Selection.Find.ClearFormatting()
    w.Selection.Find.Replacement.ClearFormatting()
    w.Selection.Find.Execute(OldStr, False, False, False, False, False, True, 1, True, NewStr, 2)
    doc.Close()
page = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.docx' in file:
            file_dir = roots+'/'+file
            file_dir = file_dir.replace('/','\\')
            Text_chang(file_dir,OldStr,NewStr)
            i = 1+1
print(i)
