import numpy as np
from gurobipy import *
import datetime

import sys
#sys.path.append('/cm/shared/engaging/gurobi/gurobi652/linux64/lib/python2.7')





#SVM with sparsity and l1 norm constraint

def Gurobi_SVM(type_loss, type_penalization, l0_OR_not, X, y, K, alpha, time_limit, model, warm_start):

    
#TYPE_LOSS = HINGE LOSS or SQUARED HINGE LOSS

#TYPE_PENALIZATION = L1 or L2

#l0_OR_not = L0 or NOT L0


#POSSIBILITY TO USE THE PREVIOUS MODEL
    



#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P = X.shape

    if(model == 0):
    
    #---VARIABLES
        Gurobi_SVM=Model("Gurobi_SVM")
        Gurobi_SVM.setParam('TimeLimit', time_limit)

        beta = np.array([Gurobi_SVM.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
        b0 = Gurobi_SVM.addVar(lb=-GRB.INFINITY, name="b0")
        
        #Sparsity constraint
        z = [Gurobi_SVM.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]
        
        #Absolute values
        abs_beta = np.array([Gurobi_SVM.addVar(lb=0, name="abs_beta_"+str(i)) for i in range(P)])
        
        #Hinge loss
        u = np.array([Gurobi_SVM.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])
        
        Gurobi_SVM.update()
        


    #---OBJECTIVE VALUE WITH HINGE LOSS AND L1-NORM
        dict_loss = {'hinge': quicksum(u),
                     'squared_hinge': quicksum(u[i]*u[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(abs_beta),
                             'l2': quicksum(beta[i]*beta[i] for i in range(P))}
    
        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)


    #---CONSTRAINTS
        
        
        #Absolute value constraints
        for i in range(P):
            Gurobi_SVM.addConstr(abs_beta[i] >= beta[i], name='abs_beta_sup_'+str(i))
            Gurobi_SVM.addConstr(abs_beta[i] >= -beta[i], name='abs_beta_inf_'+str(i))

        #Max constraint
        for i in range(N):
            Gurobi_SVM.addConstr(u[i] >= 1-y[i]*(b0 + quicksum([X[i][k]*beta[k] for k in range(P)])), 
                                name="slack_"+str(i))
        

        
        #L0 CONSTRAINT OR NOT
        if l0_OR_not=='l0':

            #Sparsity and bound constraints
            Gurobi_SVM.addConstr(quicksum(z) <= K, "sparsity")

            M=5*K
            for i in range(P):
                Gurobi_SVM.addConstr(abs_beta[i] <= M*z[i], "max_beta_"+str(i))

            
        
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        Gurobi_SVM = model.copy()
        
    #---Change the objective
        u = [Gurobi_SVM.getVarByName('loss_'+str(i)) for i in range(N)]
        beta = [Gurobi_SVM.getVarByName('beta_'+str(i)) for i in range(P)]
        abs_beta = [Gurobi_SVM.getVarByName('abs_beta_'+str(i)) for i in range(P)]
    
        dict_loss = {'hinge': quicksum(u),
                     'squared_hinge': quicksum(u[i]*u[i] for i in range(N))}

        dict_penalization = {'l1': quicksum(abs_beta),
                             'l2': quicksum(beta[i]*beta[i] for i in range(P))}
        
        Gurobi_SVM.setObjective(dict_loss[type_loss] + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)
        
        
    #---Change the sparsity constraint
        if l0_OR_not=='l0':
            z = [Gurobi_SVM.getVarByName('z_'+str(i)) for i in range(P)]
            
            constraint = Gurobi_SVM.getConstrByName('sparsity')
        
        
        #---Check if exisiting constraint
            if constraint != None:
                Gurobi_SVM.remove(constraint)
            else:
                M=5*K
                for i in range(P):
                    Gurobi_SVM.addConstr(abs_beta[i] <= M*z[i], "max_beta_"+str(i))
               
            
            Gurobi_SVM.addConstr(quicksum(z) <= K, 'sparsity')
        


        


#---FOR SECTION 3 WE HAVE NO MODEL BUT A WARM START
    if(warm_start != 0):

        (beta_start, b0_start) = warm_start

        b0.start = b0_start

        for i in range(P):
            beta[i].start = beta_start[i]
            z[i].start = int(beta_start[i]!=0)
            abs_beta[i].start = abs(beta_start[i])




   
#---SOLVE
    Gurobi_SVM.optimize()
                    
    beta_Gurobi_SVM = [Gurobi_SVM.getVarByName("beta_"+str(i)).x for i in range(P)]
    b0_Gurobi_SVM = Gurobi_SVM.getVarByName("b0").x
    
    model_status = Gurobi_SVM.status
    if (Gurobi_SVM.status == GRB.Status.OPTIMAL):
        model_status = 'Optimal'
    elif (Gurobi_SVM.status == GRB.Status.TIME_LIMIT):
        model_status = 'Time Limit'
        

    
    
    return np.round(beta_Gurobi_SVM,8) , np.round(b0_Gurobi_SVM,8), Gurobi_SVM, model_status, Gurobi_SVM


    






