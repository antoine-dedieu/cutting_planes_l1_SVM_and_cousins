import time
import numpy as np
from gurobipy import *

from group_Linf_SVM_ColG_model import *



#-----------------------------------------SOLVES GROUP LINF_SVM WITH COLUMN GENERATION WHEN P>>N--------------------------------------------------

def group_Linf_SVM_ColG(X, y, group_to_feat, idx_groups, llambda, epsilon_RC=0, model=0):

####### INPUT #######

# X, y                : design matrix and prediction vector
# group_to_feat       : for each index of a group we have access to the list of features belonging to this group
# idx_groups          : features to to include in the model 
# llambda               : regularization coefficient  
# epsilon_RC  		  : threshold for column generation
# model        	      : previous model to speed up

####### OUTPUT #######

# model : model after column generation
	
	start_time    = time.time()
	N,P           = X.shape
	number_groups = group_to_feat.shape[0]	
	use_ColG      = len(idx_groups) < number_groups



### Build the model
	model = group_Linf_SVM_ColG_model(X, y, group_to_feat, idx_groups, llambda, model)

	
	groups_to_check = list(set(range(number_groups))-set(idx_groups))
	ones_NG = np.ones(number_groups)
	


### Infinite loop until all the variables have reduced cost higher than threshold
	loop=0
	print '\n# 2/ Run Column Generation'
	print 'Number of groups in model: '+str(len(idx_groups))

	while True:
		loop += 1
		print 'Loop '+str(loop)
		
	### Solve the model
		model.optimize()


	### CASE 1: use column generation
		if use_ColG:

		### Get the dual variables
			slacks_vars        = [model.getConstrByName('slack_'+str(i)) for i in range(N)]
			slacks_vars_values = [slacks_var.Pi for slacks_var in slacks_vars]
			

		### Compute all reduce costs of the groups not in the model
			RC_aux   = np.array([y[i]*slacks_vars_values[i] for i in range(N)])
			RC_array = []
			for idx in groups_to_check: RC_array.append(llambda - np.sum( np.abs(np.dot(X[:, np.array(group_to_feat[idx])].T, RC_aux))) )

			idx_groups_to_add= np.array(groups_to_check)[np.array(RC_array) < -epsilon_RC]
					   

		### Add features with reduced costs lower than threshold
			n_groups_to_add = idx_groups_to_add.shape[0]

			if n_groups_to_add>0:
				print 'Number of groups added: '+str(n_groups_to_add)

				model = add_groups_Linf_SVM(X, y, group_to_feat, model, idx_groups_to_add, slacks_vars, llambda) 

				for group_to_add in idx_groups_to_add:
					idx_groups.append(group_to_add)
					groups_to_check.remove(group_to_add)

			else:
				break

			
	### CASE 2: break
		else:
			break 


### Solution
	betas_plus  = []
	betas_minus = []
	for idx in idx_groups:
		betas_plus   += [model.getVarByName('beta_+_'+str(j)).X  for j in group_to_feat[idx]]
		betas_minus  += [model.getVarByName('beta_-_'+str(j)).X  for j in group_to_feat[idx]]

	beta = np.array(betas_plus) - np.array(betas_minus)

	print '\nLen support = '+str(len(np.where(beta!=0)[0]))


### Obj val
	obj_val = model.ObjVal
	print 'Obj value   = '+str(obj_val)

### Time stops
	time_ColG = time.time()-start_time
	print 'Time ColG = '+str(round(time_ColG, 3))
	
	return model









