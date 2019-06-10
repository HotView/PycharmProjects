# input 不会把换行符读入输入的文件对象
# sys.stdin会把输入的换行符输入文件对象
import sys

a=input()

stdin = sys.stdin.read()



