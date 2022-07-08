
import copy

a= [
    [0,1,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,1,1]
]

#b= list(a)
for i in range(len(a)):
    for j in range(len(a[0])):
        #print(i,j)
        temp=a[i][j]
        if temp==0:
            #if(inde)
            index1=i
            index2=i
            while temp==0:
                if index1+1<len(a):
                    index1+=1
                else:
                    index1=len(a)-1
                if index2-1>0:
                    index2-=1
                else:
                    index2=0

                #print("ind1 :",index1)
                #print("ind2 :",index2)
                # if index2 >len(a)-1:
                #     index2=len(a)-1
                if (a[index1][j]==1):
                    a[i][j]=index1-i
                    break

                if(a[index2][j]==1):
                     a[i][j]=i-index2
                     break




print("a",a.index(min(a)))


"""Maara nee jeichita......! soon u willc ry :)"""
#print("B",b)