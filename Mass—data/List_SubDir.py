import os
test_dir  = 'E:/My EndNote Library/test'
for i  in os.listdir(test_dir):
    print("test")

def list_SubDir(dir_list):
    'for 迭代为空的话，怎么这次的for循环会自动跳过，不用进行判断'
    for f in os.listdir(dir_list):
        _f=  dir_list+'/'+f
        if os.path.isdir(_f):
            list_SubDir(_f)
        else:
            print('##')
            if  os.path.isfile(_f):
                print(_f)


dir_list = 'E:/My EndNote Library'
#print(os.listdir(dir_list))


#list_SubDir(dir_list)
#print(os.listdir(dir_list))