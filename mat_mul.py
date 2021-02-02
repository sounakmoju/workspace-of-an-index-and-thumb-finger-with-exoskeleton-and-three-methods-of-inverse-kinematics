import numpy as np
import sympy as sp


def mat_mul1(X,Y,m,l):
    #X=np.empty((m,n))
    #Y=np.empty((n,l))
    r=np.zeros((m,l))
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                r[i][j]+=X[i][k]*Y[k][j]
                #print(r)
    return r
    #print(r])
    
X = [[12,0,0],
    [0 ,5,0],
    [0 ,0,9]]

#Y =# [[5,8,1,2],
    #[6,7,3,0],
    #[4,5,9,1]]
#print(r)

print(mat_mul1(X,X,3,3))

#r=zeros((len(X),len(Y[0]))
        
