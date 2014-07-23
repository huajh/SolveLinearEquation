import numpy as np
from scipy import sparse

def conj_grad(A,b,max_iter=1000,tol=1.0e-9):
    
    A = sparse.lil_matrix(A)
    
    dim = len(b)
    x = np.zeros([dim,1])
    r = b - A*x
    p = r.copy()
    r2old = np.dot(r.T,r)
    for i in range(max_iter):
        Ap = A*p
        alpha = r2old/sum(p*Ap)        
        x = x + alpha*p
        r = b - A*x
        r2new =  np.dot(r.T,r)
       # print np.sqrt(r2new)
        if(np.sqrt(r2new)) < tol:
            break
        else:
            beta = r2new/r2old            
            p = r + beta*p
            r2old=r2new      
    return x,i