import numpy as np

def Gau_Elim(A,b0):
    N,M = A.shape 
    augMat = np.zeros((N,N+1))
    dim = len(b0)
    b = np.array(b0).reshape(dim,1)
    augMat[:,:-1] = A
    augMat[:,-1:] = b       
    #eliminate columns       
    for j in range(N): # col
        
        for i in range(j+1, N): # row
            factor = -(augMat[i,j] / augMat[j,j])
            augMat[i] = factor * augMat[j] + augMat[i]
    #now backsolve by substitution
    x = []
             
    for i in range(N):
        ii = N-1-i
        if ii==N-1:
            x.append(augMat[ii][-1] / augMat[ii][-2])
        else:
            residual = 0
            #substitute in all known coefficients
            for j in range(i):
                residual += (x[j]*augMat[ii][-2-j])
            #the equation is now reduced to ax + b = c form
            #solve with (c - b) / a
            x.append((augMat[ii][-1]-residual)/augMat[ii][-i-2])  
            
    x.reverse()   
    return np.array(x).reshape(dim,1)