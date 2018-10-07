import numpy as np
from gurobipy import *
import time

from L1_SVM_ColConG_model import *
from L1_SVM_ColG_model    import *
from L1_SVM_ConG_model    import *


#-----------------------------------------SOLVES L1_SVM WITH COLUMN AND CONSTRAINT GENERATION WHEN BOTH N AND P ARE LARGE--------------------------------------------------


def L1_SVM_ColConG(X, y, idx_samples, idx_columns, llambda, epsilon_RC=0):

####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_samples : samples  used to initialize the model 
# idx_columns : features used to initialize the model 
# llambda     : regularization coefficient   
# epsilon_RC  : threshold for reduced cost


####### OUTPUT #######

# model : model after column generation


	start_time = time.time()
   
	N,P   = X.shape
	N_CP  = len(idx_samples)
	P_CG  = len(idx_columns)

	use_ConColG = (N_CP < N) or (P_CG < P)

	print '\n# 2/ Run Column Generation'
	print 'Number of samples in model: '+str(N_CP)+' Number of features: '+str(P_CG)

### Build the model
	model = L1_SVM_ColConG_model(X, y, idx_samples, idx_columns, llambda)

### Columnsand constraints to check
	columns_to_check    = list(set(range(P))-set(idx_columns))
	constraint_to_check = list(set(range(N))-set(idx_samples))

	ones_P = np.ones(P)
	ones_N = np.ones(N)



### Infinite loop until all the columns AND constraints dont have a reduced cost higher than the threshold 
	loop = 0 
	continue_loop = True

	while continue_loop:
		loop += 1
		continue_loop = False
		print 'Loop '+str(loop)

	### Solve the model
		model.optimize()
		
	### CASE 1: use column and constraint generation
		if use_ConColG:

		### Get the dual variables
			dual_slacks       = [model.getConstrByName('slack_'+str(idx)) for idx in idx_samples]
			dual_slack_values = [dual_slack.Pi for dual_slack in dual_slacks]

		### Get the coefficient variables
			betas_plus        = [model.getVarByName('beta_+_'+str(idx)) for idx in idx_columns]
			betas_minus       = [model.getVarByName('beta_-_'+str(idx)) for idx in idx_columns]
			beta              = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
			
			b0                = model.getVarByName('b0')
			b0_value          = b0.X 

		### Compute the reduced costs of the columns not in the model
			RC_aux           = np.array([y[idx_samples[i]]*dual_slack_values[i] for i in range(N_CP)])
			X_reduced        = X[np.array(idx_samples),:][:,np.array(columns_to_check)]

			RC_array         = llambda*ones_P[len(columns_to_check)] - np.abs( np.dot(X_reduced.T, RC_aux) ) 
			violated_columns = np.array(columns_to_check)[RC_array < -epsilon_RC]
	

		### Compute all reduced costs of the constraints not in the model
			X_reduced            = X[:, np.array(idx_columns)][np.array(constraint_to_check), :]
			RC_aux               = np.dot(X_reduced, beta) + b0_value*ones_N[:N-N_CP] 
			RC_array             = ones_N[:N-N_CP] - y[np.array(constraint_to_check)]*RC_aux
			violated_constraints = np.array(constraint_to_check)[RC_array > epsilon_RC]

				   
		### Add columns with reduced costs lower than threshold
			n_columns_to_add = violated_columns.shape[0]
			P_CG += n_columns_to_add

			if n_columns_to_add>0:
				continue_loop = True
				print 'Number of columns added: '+str(n_columns_to_add)

				model = add_columns_L1_SVM(X, y, model, violated_columns, idx_samples, dual_slacks, llambda) 

				for violated_column in violated_columns:
					idx_columns.append(violated_column)
					columns_to_check.remove(violated_column)

					betas_plus.append(model.getVarByName('beta_+_'+str(violated_column)))
					betas_minus.append(model.getVarByName('beta_-_'+str(violated_column)))

			
		### Add the constraints with negative reduced costs
			n_constraints_to_add = violated_constraints.shape[0]
			N_CP += n_constraints_to_add

			if n_constraints_to_add>0:
				continue_loop = True
				print 'Number of constraints added: '+str(n_constraints_to_add)

				model = add_constraints_L1_SVM(X, y, model, violated_constraints, betas_plus, betas_minus, b0, idx_columns) 

				for violated_constraint in violated_constraints:
					idx_samples.append(violated_constraint)
					constraint_to_check.remove(violated_constraint)


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
	print 'Time ColG = '+str(round(time_ColG,2))
	
	return model








