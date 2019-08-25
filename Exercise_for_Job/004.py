n,m,k= list(map(int,input().split()))
s1 = input()
s2 = input()
res= 1
same =0
diff  =0
for x,y in zip(s1,s2):
    if x==y:
        same+=1
    else:
        diff+=1
print(pow(same,2))
