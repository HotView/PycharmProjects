n,m = list(map(int,input().split()))
amount = list(map(int,input().split()))
hash_ = dict()
for i in range(len(amount)):
    hash_[amount[i]] = i
value = list(map(int,input().split()))

res= min(amount)
for x in range()