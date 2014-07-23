# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:45:53 2014

@author: huajh
"""
#
# solve Ax=b using Gaussian Elimination
# augMat = [A | b]
#
import csv
import time  
import numpy as np
from scipy import sparse
from gau_elim import *
from conj_grad import *
 
def simple_test():
    # examples from 
    # http://en.wikipedia.org/wiki/Gaussian_elimination
    
    # 2x + y - z = 8.0
    #-3x - y + 2z = -11.0
    #-2x + y + 2z = -3.0
    
    b = [8.0,-11.0,-3.0]
    A = [[2.0,1.0,-1.0],
         [-3.0,-1.0,2.0],
         [-2.0,1.0,2.0]]
    b = np.array([[8.0],
                  [-11.0],
                  [-3.0]])
    A = np.array([[2.0,1.0,-1.0],
         [-3.0,-1.0,2.0],
         [-2.0,1.0,2.0]])
         
    print 'True: [2.0, 3.0, -1.0]'        
    print 'Gaussian elimination: '
    starttime = time.clock()
    print Gau_Elim(A,b)    
    endtime = time.clock()
    print ('running time %.6f',endtime-starttime) 
    
    b = np.array([[8.0],
                  [-11.0],
                  [-3.0]])
    A = np.array([[2.0,1.0,-1.0],
         [-3.0,-1.0,2.0],
         [-2.0,1.0,2.0]])
    
    A1 = sparse.lil_matrix(np.dot(A.T,A))    
    b1 = np.dot(A.T,b)
    #print linalg.cg(A1,b1)[0]
    print 'conjugate gradient: '
    starttime = time.clock()
    print conj_grad(A1,b1).T[0]   
    endtime = time.clock()
    print ('running time %.6f',endtime-starttime) 
    
def sparse_large_matrix_test():
    rows,cols = 20000,1000
    A = sparse.lil_matrix((rows,cols))    
    starttime = time.clock()
    csvreader = csv.reader(open('A_sparse_matrix.csv'))
    for line in csvreader:
        A[line[1],line[2]] = line[0]    
    b = np.genfromtxt('b_vector.csv',delimiter=',')
    endtime = time.clock()
    print ('loading time %.6f',endtime-starttime)    
    
    
if __name__ == '__main__':
       
    #simple_test()
    
    #sparse_large_matrix_test()
    load_data =0
    cg_test = 0
    gau_elim_test = 1
    if load_data == 1:
        rows,cols = 20000,1000
        A = sparse.lil_matrix((rows,cols))
        b = np.array([cols,1])
        starttime = time.clock()
        csvreader = csv.reader(open('A_sparse_matrix.csv'))
        for line in csvreader:
            A[int(line[1]),int(line[2])] = line[0]    
        b = np.genfromtxt('b_vector.csv',delimiter=',')    
        endtime = time.clock()
        print ('loading time %.6f',endtime-starttime)  
    
    if cg_test == 1:
        A1 = sparse.lil_matrix(A.T*A)          
        bb = np.reshape(b,(20000,1))
        b1 = A.T*bb
        print 'conjugate gradient: '
        starttime = time.clock()
        ans_cg,iters = conj_grad(A1,b1)
        print ans_cg.T[0]
        endtime = time.clock()
        
        print ('running time %.6f',endtime-starttime) 
        print iters
    if gau_elim_test == 1:        
        A2 = A.T*A       
        bb = np.reshape(b,(20000,1))
        b2 = A.T*bb        
        print 'Gaussian elimination: '
        starttime = time.clock()
        print Gau_Elim(A2.toarray(),b2)    
        endtime = time.clock()
        print ('Gaussian elimination runing time %.6f',endtime-starttime)  
        
            