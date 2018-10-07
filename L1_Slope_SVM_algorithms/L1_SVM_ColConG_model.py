import numpy as np
from gurobipy import *




def L1_SVM_ColConG_model(X, y, idx_samples, idx_columns, llambda):

####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_samples : samples  to to include in the model 
# idx_columns : features to to include in the model 
# llambda     : regularization coefficient  


####### OUTPUT #######

# L1_SVM_ColConG : new model

	
	N,P  = X.shape
	N_CP = len(idx_samples)
	P_CG = len(idx_columns)


### VARIABLES
	L1_SVM_ColConG=Model("L1_SVM_CG_CP")
	L1_SVM_ColConG.setParam('OutputFlag', False )
	
	#Hinge loss
	xi = np.array([L1_SVM_ColConG.addVar(lb=0, name="loss_"+str(idx_samples[i])) for i in range(N_CP)])

	#Beta
	beta_plus  = np.array([L1_SVM_ColConG.addVar(lb=0, name="beta_+_"+str(idx_columns[i])) for i in range(P_CG)])
	beta_minus = np.array([L1_SVM_ColConG.addVar(lb=0, name="beta_-_"+str(idx_columns[i])) for i in range(P_CG)])
	b0 = L1_SVM_ColConG.addVar(lb=-GRB.INFINITY, name="b0")
	
	L1_SVM_ColConG.update()


### OBJECTIVE VALUE 
	L1_SVM_ColConG.setObjective(quicksum(xi) + llambda*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


### HIGE CONSTRAINTS
	for i in range(N_CP):
		L1_SVM_ColConG.addConstr(xi[i] + y[idx_samples[i]]*(b0 + quicksum([ X[idx_samples[i]][idx_columns[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(idx_samples[i]))


	L1_SVM_ColConG.update()    
	return L1_SVM_ColConG








