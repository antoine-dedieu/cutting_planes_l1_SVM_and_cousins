import numpy as np
from gurobipy import *




#-----------------------------------------DEFINE MODEL FOR COLUMN GENERATION WITH GROUP SVM--------------------------------------------------

def group_Linf_SVM_ColG_model(X, y, group_to_feat, idx_groups, llambda, group_Linf_SVM_ColG):

####### INPUT #######

# X, y                : design matrix and prediction vector
# group_to_feat       : for each index of a group we have access to the list of features belonging to this group
# idx_groups          : indexes of groups used to initialize the model 
# llambda             : regularization coefficient  
# group_Linf_SVM_ColG : previous model to speed up


####### OUTPUT #######

# group_Linf_SVM_ColG : new model
	

	N,P      = X.shape
	n_groups = len(idx_groups)


### CASE 1: Define a new model if no one
	if group_Linf_SVM_ColG==0:
		group_Linf_SVM_ColG=Model("group_Linf_SVM_CG")
		group_Linf_SVM_ColG.setParam('OutputFlag', False )
		
	### VARIABLES
		
		#Hinge loss
		xi = np.array([group_Linf_SVM_ColG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)]) 

		#Linf and features of each groups in the model
		max_groups = []       
		for idx in idx_groups:
			max_groups.append( group_Linf_SVM_ColG.addVar(lb=0, name="max_group_"+str(idx)) )
			for j in group_to_feat[idx]:
				group_Linf_SVM_ColG.addVar(lb=0, name="beta_+_"+str(j)) 
				group_Linf_SVM_ColG.addVar(lb=0, name="beta_-_"+str(j)) 
		
		b0 = group_Linf_SVM_ColG.addVar(lb=-GRB.INFINITY, name="b0")

		group_Linf_SVM_ColG.update()


	### OBJECTIVE VALUE 
		group_Linf_SVM_ColG.setObjective(quicksum(xi) + llambda*quicksum(max_groups[i] for i in range(n_groups)), GRB.MINIMIZE)


	### HIGE CONSTRAINTS
		for i in range(N):
			group_Linf_SVM_ColG.addConstr(xi[i] + y[i]*(b0 + quicksum([ quicksum([ X[i][j]*(group_Linf_SVM_ColG.getVarByName('beta_+_'+str(j)) - group_Linf_SVM_ColG.getVarByName('beta_-_'+str(j)) )
				for j in group_to_feat[idx]])  for idx in idx_groups]) )>= 1, name="slack_"+str(i))


	### INFINITE NORM CONSTRAINTS ON GROUPS
		for idx in idx_groups:
			max_group  = group_Linf_SVM_ColG.getVarByName('max_group_'+str(idx))

			for j in group_to_feat[idx]:    
				beta_plus  = group_Linf_SVM_ColG.getVarByName('beta_+_'+str(j))
				beta_minus = group_Linf_SVM_ColG.getVarByName('beta_-_'+str(j))
				group_Linf_SVM_ColG.addConstr( max_group - beta_plus - beta_minus >= 0, name="group_"+str(idx)+"feat_"+str(j))


		
### CASE 2: Update previous model
	else:
		xi          = [group_Linf_SVM_ColG.getVarByName('loss_'+str(i)) for i in range(N)]
		max_groups  = [group_Linf_SVM_ColG.getVarByName('max_group_'+str(idx)) for idx in idx_groups]

		group_Linf_SVM_ColG.setObjective(quicksum(xi) + llambda*quicksum(max_groups[i] for i in range(n_groups)), GRB.MINIMIZE)
   
	return group_Linf_SVM_ColG







#-----------------------------------------------ADD GROUPS TO MODEL--------------------------------------------------


def add_groups_group_Linf_SVM(X, y, group_to_feat, group_Linf_SVM_ColG, idx_groups_to_add, slacks_vars, llambda):

####### INPUT #######

# X, y                : design matrix and prediction vector
# group_to_feat       : for each index of a group we have access to the list of features belonging to this group
# group_Linf_SVM_ColG : new model
# idx_groups_to_add   : indexes of groups to add
# slacks_vars         : slacks variables
# llambda			  : regularization parameter

####### OUTPUT #######

# group_Linf_SVM_ColG : model updated
	

### Loop over new groups
	for idx in idx_groups_to_add:
	
		max_group = group_Linf_SVM_ColG.addVar(lb=0, obj = llambda, name="max_group_"+str(idx))

	### Loop over features in the group
		for j in group_to_feat[idx]:
			col_plus, col_minus = Column(), Column()
			for i in range(N):
				col_plus.addTerms(  y[i]*X[i][j], cstrts_slacks[i]) #coeff for constraint, name of constraint
				col_minus.addTerms(-y[i]*X[i][j], cstrts_slacks[i])
				
		### New columns
			beta_plus  = group_Linf_SVM_ColG.addVar(lb=0, obj = 0, column=col_plus,  name="beta_+_"+str(j) )
			beta_minus = group_Linf_SVM_ColG.addVar(lb=0, obj = 0, column=col_minus, name="beta_-_"+str(j) )
			group_Linf_SVM_ColG.update()


		### Infinite norm constraint on the group
			group_Linf_SVM_ColG.addConstr( max_group - beta_plus - beta_minus >= 0, name="group_"+str(idx)+"feat_"+str(j))
			group_Linf_SVM_ColG.update()
			

	return group_Linf_SVM_ColG





