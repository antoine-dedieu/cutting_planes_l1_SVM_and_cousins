import numpy as np
from gurobipy import *



#-----------------------------------------BUILD MODEL FOR CONSTRAINT GENERATION ONLY--------------------------------------------------

def L1_SVM_ConG_model(X, y, idx_samples, llambda):

####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_samples : constraints to include in the model 
# llambda       : regularization coefficient  


####### OUTPUT #######

# L1_SVM_ConG : new model
	

	N,P  = X.shape
	N_CP = len(idx_samples)

	
	
### VARIABLES
	L1_SVM_ConG=Model("L1_SVM_CP")
	L1_SVM_ConG.setParam('OutputFlag', False )
	
	#Hinge loss
	xi = np.array([L1_SVM_ConG.addVar(lb=0, name="loss_"+str(idx_samples[i])) for i in range(N_CP)])

	#Features
	beta_plus  = np.array([L1_SVM_ConG.addVar(lb=0, name="beta_+_"+str(i)) for i in range(P)])
	beta_minus = np.array([L1_SVM_ConG.addVar(lb=0, name="beta_-_"+str(i)) for i in range(P)])
	b0 = L1_SVM_ConG.addVar(lb=-GRB.INFINITY, name="b0")
	L1_SVM_ConG.update()


### OBJECTIVE VALUE 
	L1_SVM_ConG.setObjective(quicksum(xi) + llambda*quicksum(beta_plus[i]+beta_minus[i] for i in range(P)), GRB.MINIMIZE)


### HIGE CONSTRAINTS ONLY FOR SUBSET
	for i in range(N_CP):
		L1_SVM_ConG.addConstr(xi[i] + y[idx_samples[i]]*(b0 + quicksum([ X[idx_samples[i]][k]*(beta_plus[k] - beta_minus[k]) for k in range(P)]))>= 1, 
							 name="slack_"+str(idx_samples[i]))
  
	L1_SVM_ConG.update()
	return L1_SVM_ConG







#-----------------------------------------------ADD COLUMNS--------------------------------------------------

def add_constraints_L1_SVM(X, y, L1_SVM_ConG, violated_constraints, betas_plus, betas_minus, b0, idx_columns):

####### INPUT #######

# L1_SVM_ConG      		: model
# violated_columns 		: constraints to add to the model
# beta_plus, beta_minus : coefficients variables
# llambda               : regularization parameter
# idx_columns     		: columns in the model (different from range(P) when P is large)


####### OUTPUT #######

# L1_SVM_ConG: model updated

	for violated_constraint in violated_constraints:

		#New variable
		xi_violated = L1_SVM_ConG.addVar(lb=0, obj = 1, column=Column(),  name="loss_"+str(violated_constraint) )
		L1_SVM_ConG.update()

		#New constraint
		L1_SVM_ConG.addConstr( xi_violated + y[violated_constraint]*(b0 + quicksum([ X[violated_constraint][idx_columns[k]]*(betas_plus[k] - betas_minus[k]) 
								for k in range(len(idx_columns))]))>= 1, name="slack_"+str(violated_constraint))
        
	L1_SVM_ConG.update()    

	return L1_SVM_ConG





