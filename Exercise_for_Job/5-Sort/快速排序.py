a= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
def quick_sort(array,left,right):
    if left>=right:
        return
        # return array
    key = array[left]
    low = left
    high = right
    # 循环参考点直到遍历全部
    while left<right:
        while left<right and array[right]>=key:
            right-=1
        array[left] = array[right]
        while left<right and array[left]<=key:
            left+=1
        array[right] = array[left]
    array[left] = key
    quick_sort(array,low,left-1)
    quick_sort(array, left+1, high)
    return array
array = quick_sort(a,0,len(a)-1)
print(a)

