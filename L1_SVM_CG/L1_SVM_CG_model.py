import numpy as np
from gurobipy import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------BUILD THE MODEL--------------------------------------------------

def L1_SVM_CG_model(X, y, idx_CG, alpha, time_limit, model, f):

#idx_CG    = index of the features used to generate 
#MODEL     = previous model to speed up
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    P_CG = len(idx_CG)

    write_and_print('Size of model: '+str([N, P_CG]), f)

    
    if(model == 0):
    
    #---VARIABLES
        L1_SVM_CG=Model("L1_SVM_CG")
        L1_SVM_CG.setParam('OutputFlag', False )
        L1_SVM_CG.setParam('TimeLimit', time_limit)
        
        
        #Hinge loss
        xi = np.array([L1_SVM_CG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        #Beta -> name correspond to real index
        beta_plus  = np.array([L1_SVM_CG.addVar(lb=0, name="beta_+_"+str(idx_CG[i])) for i in range(P_CG)])
        beta_minus = np.array([L1_SVM_CG.addVar(lb=0, name="beta_-_"+str(idx_CG[i])) for i in range(P_CG)])
        b0 = L1_SVM_CG.addVar(lb=-GRB.INFINITY, name="b0")
        
        L1_SVM_CG.update()


    #---OBJECTIVE VALUE 
        L1_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            L1_SVM_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_CG[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, 
                                 name="slack_"+str(i))
      
    #---RELAX
        L1_SVM_CG.update()
        
    
    
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        L1_SVM_CG = model.copy()
        
        xi          = [L1_SVM_CG.getVarByName('loss_'+str(i)) for i in range(N)]
        beta_plus   = [L1_SVM_CG.getVarByName('beta_+_'+str(idx_CG[i])) for i in range(P_CG)]
        beta_minus  = [L1_SVM_CG.getVarByName('beta_-_'+str(idx_CG[i])) for i in range(P_CG)]

        L1_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)
   

    L1_SVM_CG.update()
    
    return L1_SVM_CG







#-----------------------------------------------ADD VARIABLES--------------------------------------------------

def add_column_L1_SVM(X, y, L1_SVM_CG, violated_column, idx_samples, alpha):
#idx_samples: samples in the model -> may be different from range(N) when deleting samples

# Add a column in a model with respective constraints
    
    N,P = X.shape
        
    col_plus, col_minus = Column(), Column()
    slack_constraints = [L1_SVM_CG.getConstrByName('slack_'+str(idx)) for idx in idx_samples]
    
    for i in range(len(idx_samples)):
        col_plus.addTerms(  y[idx_samples[i]]*X[idx_samples[i]][violated_column], slack_constraints[i])
        col_minus.addTerms(-y[idx_samples[i]]*X[idx_samples[i]][violated_column], slack_constraints[i])
        
        
#---ADD IN BOTH CASES
    beta_plus  = L1_SVM_CG.addVar(lb=0, obj = alpha, column=col_plus,  name="beta_+_"+str(violated_column) )
    beta_minus = L1_SVM_CG.addVar(lb=0, obj = alpha, column=col_minus, name="beta_-_"+str(violated_column) )

    L1_SVM_CG.update()

    return L1_SVM_CG





