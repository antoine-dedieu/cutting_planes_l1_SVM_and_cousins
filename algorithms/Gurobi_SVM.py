import numpy as np
from gurobipy import *
import datetime

import sys
#sys.path.append('/cm/shared/engaging/gurobi/gurobi652/linux64/lib/python2.7')




def Gurobi_SVM(type_loss, type_penalization, X, y, idx_CG, alpha, Gurobi_SVM, warm_start):

#idx_CG     = index of the features used to generate 
#Gurobi_SVM = previous model to speed up
#WARM START = speed up Gurobi on full model 
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    P_CG = len(idx_CG)

    
    if(Gurobi_SVM == 0):
    
    #---VARIABLES
        Gurobi_SVM=Model("L1_SVM_CG")
        Gurobi_SVM.setParam('OutputFlag', False )
        Gurobi_SVM.setParam('TimeLimit', 60)
        
        
        #Hinge loss
        xi = np.array([Gurobi_SVM.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        beta_plus  = np.array([Gurobi_SVM.addVar(lb=0, name="beta_+_"+str(idx_CG[i])) for i in range(P_CG)])
        beta_minus = np.array([Gurobi_SVM.addVar(lb=0, name="beta_-_"+str(idx_CG[i])) for i in range(P_CG)])
        b0 = Gurobi_SVM.addVar(lb=-GRB.INFINITY, name="b0")

        Gurobi_SVM.update()


    #---OBJECTIVE VALUE WITH HINGE LOSS AND L1-NORM
        dict_loss = {'hinge': quicksum(xi),
                     'squared_hinge': quicksum(xi[i]*xi[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(  beta_plus[i]+beta_minus[i]                               for i in range(P_CG)),
                             'l2': quicksum( (beta_plus[i]+beta_minus[i])*(beta_plus[i]+beta_minus[i]) for i in range(P_CG))}
    
        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            Gurobi_SVM.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_CG[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, 
                                 name="slack_"+str(i))


    #---POSSIBLE WARM START (only for Gurobi on full model)
        if len(warm_start) > 0:

            for i in range(P_CG):
                beta_plus[i].start  = max( warm_start[idx_CG[i]], 0)
                beta_minus[i].start = max(-warm_start[idx_CG[i]], 0)


           
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        #L1_SVM_CG = model.copy()
        xi          = [Gurobi_SVM.getVarByName('loss_'+str(i)) for i in range(N)]
        beta_plus   = [Gurobi_SVM.getVarByName('beta_+_'+str(idx_CG[i])) for i in range(P_CG)]
        beta_minus  = [Gurobi_SVM.getVarByName('beta_-_'+str(idx_CG[i])) for i in range(P_CG)]

    #---OBJECTIVE VALUE WITH HINGE LOSS AND L1-NORM
        dict_loss = {'hinge':         quicksum(xi),
                     'squared_hinge': quicksum(xi[i]*xi[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(  beta_plus[i]+beta_minus[i]                               for i in range(P_CG)),
                             'l2': quicksum( (beta_plus[i]+beta_minus[i])*(beta_plus[i]+beta_minus[i]) for i in range(P_CG))}
    
        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)
   


#---RESULTS
    Gurobi_SVM.update()
    Gurobi_SVM.optimize()
    
    beta = np.zeros(P)
    for i in range(P_CG): beta[idx_CG[i]] = beta_plus[i].X - beta_minus[i].X

    beta_0  = Gurobi_SVM.getVarByName('b0').X 
    obj_val = Gurobi_SVM.ObjVal
    

    return beta, beta_0, obj_val, Gurobi_SVM


