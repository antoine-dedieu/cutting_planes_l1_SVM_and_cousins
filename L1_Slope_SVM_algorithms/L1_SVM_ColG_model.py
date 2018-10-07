import numpy as np
from gurobipy import *



#-----------------------------------------DEFINE MODEL FOR COLUMN GENERATION ONLY--------------------------------------------------

def L1_SVM_ColG_model(X, y, idx_columns, llambda, L1_SVM_ColG):

####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_columns : features to to include in the model 
# llambda       : regularization coefficient  
# L1_SVM_ColG : previous model to speed up


####### OUTPUT #######

# L1_SVM_ColG : new model

	N,P  = X.shape
	P_CG = len(idx_columns)


### CASE 1: Define a new model if no one
	if L1_SVM_ColG==0:

		L1_SVM_ColG=Model("L1_SVM_CG")
		L1_SVM_ColG.setParam('OutputFlag', False)


	### VARIABLES
		
		#Hinge loss
		xi = np.array([L1_SVM_ColG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

		#Features
		beta_plus  = np.array([L1_SVM_ColG.addVar(lb=0, name="beta_+_"+str(idx_columns[i])) for i in range(P_CG)])
		beta_minus = np.array([L1_SVM_ColG.addVar(lb=0, name="beta_-_"+str(idx_columns[i])) for i in range(P_CG)])
		b0 = L1_SVM_ColG.addVar(lb=-GRB.INFINITY, name="b0")

		L1_SVM_ColG.update()


	### OBJECTIVE VALUE 
		L1_SVM_ColG.setObjective(quicksum(xi) + llambda*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


	### HIGE CONSTRAINTS
		for i in range(N):
			L1_SVM_ColG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_columns[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(i))
		   
		
	
### CASE 2: Update previous model
	else:
		xi          = [L1_SVM_ColG.getVarByName('loss_'+str(i)) for i in range(N)]
		beta_plus   = [L1_SVM_ColG.getVarByName('beta_+_'+str(idx_columns[i])) for i in range(P_CG)]
		beta_minus  = [L1_SVM_ColG.getVarByName('beta_-_'+str(idx_columns[i])) for i in range(P_CG)]

		L1_SVM_ColG.setObjective(quicksum(xi) + llambda*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)

	return L1_SVM_ColG







#-----------------------------------------------ADD COLUMNS TO MODEL--------------------------------------------------

def add_columns_L1_SVM(X, y, L1_SVM_ColG, violated_columns, idx_columns, slacks_vars, llambda):

####### INPUT #######

# X, y             : design matrix and prediction vector
# L1_SVM_ColG      : model
# violated_columns : columns to add to the model
# idx_columns      : samples in the model (different from range(N)whne n is large)
# slacks_vars.     : slacks variables
# llambda          : regularization parameter


####### OUTPUT #######

# L1_SVM_ColG: model updated

	for violated_column in violated_columns:

	### Columns to add in th model for the variables added
		col_plus, col_minus = Column(), Column()

		for i, idx_column in enumerate(idx_columns):
			X_row_col = X[idx_column][violated_column]
		
			col_plus.addTerms(  y[idx_column]*X_row_col, slacks_vars[i])
			col_minus.addTerms(-y[idx_column]*X_row_col, slacks_vars[i])
			
	### Add 2 new positive columns
		beta_plus  = L1_SVM_ColG.addVar(lb=0, obj = llambda, column=col_plus,  name="beta_+_"+str(violated_column) )
		beta_minus = L1_SVM_ColG.addVar(lb=0, obj = llambda, column=col_minus, name="beta_-_"+str(violated_column) )

	L1_SVM_ColG.update()
	return L1_SVM_ColG





