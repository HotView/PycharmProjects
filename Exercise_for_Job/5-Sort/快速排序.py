a= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
def quick_sort(array,left,right):
    if left>=right:
        return
        # return array
    i = left
    j = right
    key = array[i]
    # 循环参考点直到遍历全部
    while i<j:
        while i<j and array[j]>=key:
            j-=1
        array[i] = array[j]
        while i<j and array[i]<=key:
            i+=1
        array[j] = array[i]
    array[i] = key
    quick_sort(array,left,i-1)
    quick_sort(array, i+1, right)
    return array
array = quick_sort(a,0,len(a)-1)
print(a)

