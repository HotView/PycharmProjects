def ins_sort(seq,i):
    if i==0:
        return
    ins_sort(seq,i-1)
    j = i
    while j>0 and seq[j-1]>seq[j]:
        seq[j-1],seq[j] = seq[j],seq[j-1]
        j = j-1


seq= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
#不是严格的插入排序不是
n = len(seq)

for i in range(1,n):
    j = i
    while j>0 and seq[j-1]>seq[j]:
        seq[j-1],seq[j] = seq[j],seq[j-1]
        j = j-1
print(seq)
