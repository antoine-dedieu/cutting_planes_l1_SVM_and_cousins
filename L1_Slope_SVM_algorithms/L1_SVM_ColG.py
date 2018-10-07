import time
import numpy as np
from gurobipy import *
from L1_SVM_ColG_model import *



#-----------------------------------------SOLVES L1_SVM WITH COLUMN GENERATION WHEN P>>N--------------------------------------------------


def L1_SVM_ColG(X, y, idx_columns, alpha, epsilon_RC=0, model=0):


####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_columns: columns used to initialize the model
# alpha       : regularization coefficient  
# epsilon_RC  : threshold for reduced costs
# model.      : use the previous model to speed up the computation


####### OUTPUT #######

# model : model after column generation

	start_time = time.time()

	N,P        = X.shape
	P_CG       = len(idx_columns)
	use_ColG   = P_CG < P

	print '\n# 2/ Run Column Generation'
	print  'Number of columns in model: '+str(P_CG)

### Build the model
	model = L1_SVM_ColG_model(X, y, idx_columns, alpha, model) #model=0 -> no warm start else update the objective function

### Columns outside the model
	columns_to_check = list(set(range(P))-set(idx_columns)) #columns not in model
	ones_P = np.ones(P)
	

### Infinite loop until all the columns have reduced cost higher than the threshold
	loop = 0 
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
		
		### Compute the reduced costs of the columns not in the model
			RC_aux            = np.array([y[i]*slacks_vars_values[i] for i in range(N)])
			RC_array          = alpha*ones_P[len(columns_to_check)] - np.abs( X[:, np.array(columns_to_check)].T.dot( RC_aux) )

		### Add columns with reduced costs lower than threshold
			violated_columns = np.array(columns_to_check)[RC_array < -epsilon_RC]
			n_columns_to_add = violated_columns.shape[0]

			if n_columns_to_add>0:
				print 'Number of columns added: '+str(n_columns_to_add)
				model = add_columns_L1_SVM(X, y, model, violated_columns, range(N), slacks_vars, alpha) 

				for column_to_add in violated_columns:
					idx_columns.append(column_to_add)
					columns_to_check.remove(column_to_add)

			else:
				break
			
	### CASE 2: break   
		else:
			break 


### Solution
	beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in idx_columns])
	beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in idx_columns])
	beta    	= np.array(beta_plus) - np.array(beta_minus)
	print '\nLen support = '+str(len(np.where(beta!=0)[0]))

### Obj val
	obj_val = model.ObjVal
	print 'Obj value   = '+str(obj_val)

### Time stops
	time_ColG = time.time()-start_time
	print 'Time ColG = '+str(round(time_ColG, 3))
	
	return model









