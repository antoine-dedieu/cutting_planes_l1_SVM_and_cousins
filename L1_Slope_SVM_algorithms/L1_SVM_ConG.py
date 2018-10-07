import time
import numpy as np
from gurobipy import *

from L1_SVM_ConG_model import *



#-----------------------------------------SOLVES L1_SVM WITH CONSTRAINT GENERATION WHEN N>>P--------------------------------------------------

def L1_SVM_ConG(X, y, idx_samples, llambda, epsilon_RC=0):


####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_samples : samples used to intiailize the model
# llambda     : regularization coefficient  
# epsilon_RC  : threshold for reduced cost


####### OUTPUT #######

# model : model after column generation
	
	start_time = time.time()
	
	N,P       = X.shape
	N_samples = len(idx_samples)
	use_ConG  = N_samples < N

	print '\n# 2/ Run Column Generation'
	print 'Number of samples in model: '+str(N_samples)


### Build the model
	model = L1_SVM_ConG_model(X, y, idx_samples, llambda)

### Constraints outside the model
	constraint_to_check = list(set(range(N))-set(idx_samples))
	ones_N = np.ones(N)

### Infinite loop until all the constraints have reduced costs higher than the theshold
	loop = 0
	while True: 
		loop += 1
		print 'Loop '+str(loop)
		
	### Solve the model
		model.optimize()


	### CASE 1: use constraint generation
		if use_ConG:

		### Get the coefficient variables
			betas_plus  = [model.getVarByName('beta_+_'+str(idx)) for idx in range(P)]
			betas_minus = [model.getVarByName('beta_-_'+str(idx)) for idx in range(P)]
			beta        = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
			b0          = model.getVarByName('b0')
			b0_val      = b0.X
			

		### Compute all reduced costs of the constraints not in the model
			RC_aux           = np.dot(X[np.array(constraint_to_check), :], beta) + b0_val*ones_N[:N-N_samples]
			RC_array         = ones_N[:N-N_samples] - y[np.array(constraint_to_check)]*RC_aux


		### Look for constraints with negative reduced costs
			violated_constraints = np.array(constraint_to_check)[RC_array > epsilon_RC]


		### Add the constraints with negative reduced costs
			n_constraints_to_add = violated_constraints.shape[0]
			N_samples		    += n_constraints_to_add

			if n_constraints_to_add>0:
				print 'Number of constraints added: '+str(n_constraints_to_add)
				model = add_constraints_L1_SVM(X, y, model, violated_constraints, betas_plus, betas_minus, b0, range(P)) 

				for violated_constraint in violated_constraints:
					idx_samples.append(violated_constraint)
					constraint_to_check.remove(violated_constraint)

			else:
				break
			

	### CASE 2: break
		else:
			break


### Solution
	beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in range(P)])
	beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in range(P)])
	beta        = np.array(beta_plus) - np.array(beta_minus)
	print '\nLen support = '+str(len(np.where(beta!=0)[0]))

### Obj val
	obj_val = model.ObjVal
	print 'Obj value   = '+str(obj_val)

### Time 
	time_ConG = time.time()-start_time
	print 'Time ConG = '+str(round(time_ConG, 3))

	return model









