readline = input().split()
erro_list= []
name_split = readline[0].split('\\')
filename = name_split[-1]
erro_record = [filename,readline[1],1]
erro_list.append(erro_record)
print(erro_list[0][0],erro_list[0][1],erro_list[0][2])

