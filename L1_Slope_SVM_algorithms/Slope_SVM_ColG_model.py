import numpy as np
from gurobipy import *



#-----------------------------------------DEFINE MODEL FOR SLOPE WITH COLUMN GENERATION--------------------------------------------------

def Slope_SVM_ColG_model(X, y, w_star, idx_columns):

####### INPUT #######

# X, y        : design matrix and prediction vector
# w_star      : plane used to initialize the model with a Slope constraint, of same lenght than idx_columns
# idx_columns : columns to to include in the model 


####### OUTPUT #######

# L1_SVM_ColG : new model


	N,P  = X.shape
	P_CG = len(idx_columns)


### VARIABLES
	Slope_SVM_ColG=Model("Slope_SVM_ColG")
	Slope_SVM_ColG.setParam('OutputFlag', False)
	
	#Hinge loss
	xi = np.array([Slope_SVM_ColG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

	#Beta 
	beta_plus  = np.array([Slope_SVM_ColG.addVar(lb=0, name="beta_+_"+str(idx_columns[i])) for i in range(P_CG)])
	beta_minus = np.array([Slope_SVM_ColG.addVar(lb=0, name="beta_-_"+str(idx_columns[i])) for i in range(P_CG)])
	b0 = Slope_SVM_ColG.addVar(lb=-GRB.INFINITY, name="b0")

	#Slope reg
	eta = Slope_SVM_ColG.addVar(lb=-GRB.INFINITY, name="eta")
	Slope_SVM_ColG.update()


### OBJECTIVE VALUE 
	Slope_SVM_ColG.setObjective(quicksum(xi) + eta, GRB.MINIMIZE)


### HIGE CONSTRAINTS
	for i in range(N): 
		Slope_SVM_ColG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_columns[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, name="slack_"+str(i))


### SlOPE CONSTRAINTS FOR PLANE
	Slope_SVM_ColG.addConstr( quicksum([ w_star[k]*(beta_plus[k] + beta_minus[k]) for k in range(P_CG)]) <= eta, name="w_star_0")

	return Slope_SVM_ColG





#-----------------------------------------------ADD COLUMNS TO MODEL--------------------------------------------------

def add_columns_Slope_SVM(X, y, Slope_SVM_ColG, violated_columns, violated_llambdas, slacks_vars, all_w_star_cstrts):

####### INPUT #######

# X, y        		: design matrix and prediction vector
# Slope_SVM_ColG    : model
# violated_columns  : columns to add to the model
# violated_llambdas : Slope regulairization corresponding to the new columns added 
# slacks_vars       : slacks variables
# all_w_star_cstrts : list of planes in the model


####### OUTPUT #######

# Slope_SVM_ColG: model updated
	

	N,P = X.shape
	new_betas_plus  = []
	new_betas_minus = []


### Loop over columns to add
	for j in range(len(violated_columns)):
		violated_column     = violated_columns[j]
		col_plus, col_minus = Column(), Column()

	### Hinge constraints
		for i in range(N):
			X_row_col = X[i, violated_column] 
				
			col_plus.addTerms(  y[i]*X_row_col, slacks_vars[i])
			col_minus.addTerms(-y[i]*X_row_col, slacks_vars[i])

	### Add coefficients for Slope constraints in the model
		for all_w_star_cstrt in all_w_star_cstrts:
			col_plus.addTerms( violated_llambdas[j], all_w_star_cstrt)
			col_minus.addTerms(violated_llambdas[j], all_w_star_cstrt)
			
	### Add columns to the model
		Slope_SVM_ColG.addVar(lb=0, column=col_plus,  name="beta_+_"+str(violated_column) ) 
		Slope_SVM_ColG.addVar(lb=0, column=col_minus, name="beta_-_"+str(violated_column) ) 

	Slope_SVM_ColG.update()
	return Slope_SVM_ColG




#-----------------------------------------------ADD PLANES (CONSTRAINTS)--------------------------------------------------

def add_constraints_Slope_SVM(X, y, Slope_SVM_ColG, w_star, eta, beta_plus, beta_minus, b0, P_CG, idx_cut):

####### INPUT #######

# Slope_SVM_ColG      : model
# w_star  			  : cut corresponding to the constraint to add
# eta  			      : Slope constraint
# beta_plus/minus, b0 : slacks variables
# P_CG		 		  :	number of columns in the model
# id_cut              : used for the name of the cut


####### OUTPUT #######

# Slope_SVM_ColG: model updated
	
	Slope_SVM_ColG.addConstr( quicksum([ w_star[k]*(beta_plus[k] + beta_minus[k]) for k in range(P_CG)]) <= eta, name="w_star_"+str(idx_cut))
	Slope_SVM_ColG.update()    

	return Slope_SVM_ColG




