import os, sys, re
import win32com
from win32com import client as wc
#help(win32com.client)

root_dir = sys.path[0]
OldStr = '佛山佑尔'
NewStr = '武汉宏华'
def Header_change(file_dir,OldStr,NewStr):
    w = wc.Dispatch('Word.Application')
    doc = w.Documents.Open(file_dir)
    w.ActiveDocument.Sections[0].Headers[0].Range.Find.ClearFormatting()
    w.ActiveDocument.Sections[0].Headers[0].Range.Find.Replacement.ClearFormatting()
    w.ActiveDocument.Sections[0].Headers[0].Range.Find.Execute(OldStr, False, False, False, False, False, True, 1, False, NewStr, 2)
    doc.Close()


