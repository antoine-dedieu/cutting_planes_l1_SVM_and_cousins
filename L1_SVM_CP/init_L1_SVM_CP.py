import numpy as np
from gurobipy import *

import sys
sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *


import time


def dual_L1_SVM_CP_model(X, y, idx_CP, alpha, iteration_limit):

#idx_CG    = index of the features used to generate 
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    N_CP = len(idx_CP)

    
    
#---VARIABLES
    dual_L1_SVM_CP=Model("dual_L1_SVM_CP")
    dual_L1_SVM_CP.setParam('OutputFlag', False )
    #dual_L1_SVM_CP.setParam('IterationLimit', iteration_limit)
    
    
    #Hinge loss
    pi = np.array([dual_L1_SVM_CP.addVar(lb=0, name="pi_"+str(idx_CP[i])) for i in range(N_CP)])
    dual_L1_SVM_CP.update()


#---OBJECTIVE VALUE 
    dual_L1_SVM_CP.setObjective(quicksum(pi), GRB.MINIMIZE)


#---PI CONSTRAINTS 
    for i in range(N_CP):
        dual_L1_SVM_CP.addConstr(pi[i] <= 1, name="pi_"+str(i))


#---PI CONSTRAINTS 
    for j in range(P):
        dual_L1_SVM_CP.addConstr(quicksum([ y[idx_CP[i]] * X[idx_CP[i]][j]*pi[i] for i in range(N_CP)]) <= alpha,  name="dual_beta_+_"+str(idx_CP[i]))
        dual_L1_SVM_CP.addConstr(quicksum([ y[idx_CP[i]] * X[idx_CP[i]][j]*pi[i] for i in range(N_CP)]) >= -alpha, name="dual_beta_-_"+str(idx_CP[i]))



#---ORTHOGONALITY
    dual_L1_SVM_CP.addConstr(quicksum([ pi[i]*y[idx_CP[i]] for i in range(N_CP)]) == 0, name='orthogonality')

  
#---RESULT
    dual_L1_SVM_CP.update()
    return dual_L1_SVM_CP








def restrict_lines_CP_dual(X, y, alpha, n_samples, f):
    start = time.time()
    N, P = X.shape
    iteration_limit = 1

    idx_CP       = range(N)
    n_constrains = N
    dual_L1_SVM_CP = dual_L1_SVM_CP_model(X, y, idx_CP, alpha, iteration_limit)


    while n_constrains > n_samples:

    #---Optimize model
        dual_L1_SVM_CP.optimize()
        pi = np.array([dual_L1_SVM_CP.getVarByName("pi_"+str(i)) for i in idx_CP])
        

    #---Rank constraints
        n_constraints_to_remove = min(n_constrains/2, n_constrains - n_samples)
        remove_constraints      = np.array(pi).argsort()[:n_constraints_to_remove] #pi>0
        idx_to_removes          = np.array(idx_CP)[remove_constraints]
        n_constrains           -= n_constraints_to_remove


    #---Remove constraints
        pis_to_remove = np.array([dual_L1_SVM_CP.getVarByName(name="pi_"+str(i)) for i in idx_to_removes])

        for pi_to_remove in pis_to_remove:
            dual_L1_SVM_CP.remove(pi_to_remove)
        dual_L1_SVM_CP.update()
            
        for remove_constraint in idx_to_removes:
            idx_CP.remove(remove_constraint)


    write_and_print('Time heuristic for sample subset selection: '+str(time.time()-start)+'\n', f)

    return idx_CP








def init_CP_dual(X, y, alpha, n_samples, f):
    N, P = X.shape

    #X_plus  = X[y==1]
    #X_minus = X[y==-1]

    #index_columns = RFE_for_CG(X, y, alpha, 10, f)
    index_columns = init_correlation(X, y, 10, f)

    X_RFE  = np.array([ [X[i][j] for j in index_columns] for i in range(N)])
    idx_CP = restrict_lines_CP_dual(X_RFE, y, alpha, n_samples, f)

    return idx_CP
















