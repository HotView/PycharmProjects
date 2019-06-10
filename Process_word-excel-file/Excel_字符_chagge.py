import os
import sys
from win32com import client as wc
dir_ = 'C:/Users/Q/Desktop/2017.06.15体系文件/'
OldStr ='YE'
NewStr ='HH'
def Text_chang(Excel_dir,OldStr,NewStr):
    ex = wc.Dispatch('Excel.Application')
    ex.Visible = 0
    ex.DisplayAlerts = 0
    print(Excel_dir)
    wk = ex.Workbooks.Open(Excel_dir)
    sheetname = wk.Sheets(1).Name
    ws = wk.Worksheets(sheetname)
    ws.Activate
    ex.Selection.Replace(OldStr,NewStr)
    wk.Save()
    ex.quit()
i = 0
for roots,dirs,files in os.walk(dir_):
    for file in files:
        if '.xls' in file:
                file_dir = roots+'/'+file
                file_dir = file_dir.replace('/','\\')
                Text_chang(file_dir,OldStr,NewStr)
                i = i+1
    print(i)
