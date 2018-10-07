import numpy as np
import time

from gurobipy import *
from Slope_SVM_ColG_model import *



def Slope_SVM_ColG(X, y, idx_columns, llambda_arr, beta_FOM_restricted, epsilon_RC=1e-2):

####### INPUT #######

# X, y        : design matrix and prediction vector
# idx_columns : columns to to include in the model 
# llambda_arr : sequence of Slope coefficients
# beta_FOM_restricted : FOM estimator restricted to its support
 
	start_time = time.time()
	N,P   = X.shape
	P_CG  = len(idx_columns)
	

	print '\n# 2/ Run Column Generation'
	print  'Number of columns in model: '+str(P_CG)


### Define w_start as the first Slope constraint used to initialize the model
	idx_columns_FOM_sorted = np.argsort(np.abs(beta_FOM_restricted))[::-1]
	w_star_FOM_Slope       = np.zeros(len(idx_columns))	
	for j in range(len(idx_columns)): w_star_FOM_Slope[idx_columns_FOM_sorted[j]] = llambda_arr[j]

	n_cuts = 1 #number of cuts in the model

### Build the model
	model     = Slope_SVM_ColG_model(X, y, w_star_FOM_Slope, idx_columns) 

### Columns outside the model
	columns_to_check = list(set(range(P))-set(idx_columns))
	ones_P = np.ones(P)


### Infinite loop until we add no cut or column
	loop = 0
	continue_loop = True

	while continue_loop:
		loop += 1
		continue_loop = False
		print 'Loop '+str(loop)
		
	### Solve the model
		model.optimize()
	

### Add cuts via constraint generation
	
	### Get the coefficients variables, the Slope variable, and all dual variables
		betas_plus        = [model.getVarByName('beta_+_'+str(idx)) for idx in idx_columns]
		betas_minus       = [model.getVarByName('beta_-_'+str(idx)) for idx in idx_columns]
		beta              = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
		
		b0                = model.getVarByName('b0')
		b0_value          = b0.X

		eta               = model.getVarByName('eta')
		eta_val           = eta.X

		slacks_vars        = [model.getConstrByName('slack_'+str(i)) for i in range(N)]
		slacks_vars_values = [slacks_var.Pi for slacks_var in slacks_vars]


	### Define the Slope vector associated with beta
		idx_sort     = np.argsort(np.abs(beta))[::-1]
		w_beta_Slope = np.zeros(P_CG)   
		for j in range(P_CG): w_beta_Slope[idx_sort[j]] = llambda_arr[j]
		

	### Add the cut is Slope condition is not satisfied
		if np.sum([ abs(beta[idx_sort[j]]) * llambda_arr[j] for j in range(len(idx_sort)) ]) > (1+epsilon_RC)*eta_val:
			model     = add_constraints_Slope_SVM(X, y, model, w_beta_Slope, eta, betas_plus, betas_minus, b0, P_CG, n_cuts)
			n_cuts += 1
			
			print 'Cut added'
			continue_loop = True



### Add columns via column generation

	### Get all cuts in the model
		all_w_Slope		   = [model.getConstrByName('w_star_'+str(cut)) for cut in range(n_cuts)]

	### Compute the reduced costs of the columns not in the model 
		RC_aux             = np.array([y[i]*slacks_vars_values[i] for i in range(N)])
		RC_aux             = np.abs( np.dot(X.T, RC_aux) ) 

		#Reduced costs
		RC_array           = np.array([llambda_arr[P_CG] - RC_aux[j] for j in range(P)])[columns_to_check]
		RC_argsort         = np.argsort(RC_array)
		
	### We select at most 10 columns to be added at a time 
		n_columns_to_add    = 0  
		violated_columns  = [] #idx of the column
		violated_llambdas = [] #corresponding llambda
		
		while RC_array[RC_argsort[n_columns_to_add]] < -epsilon_RC and n_columns_to_add<10: 
			violated_columns.append( columns_to_check[ RC_argsort[n_columns_to_add] ] )
			violated_llambdas.append(llambda_arr[ P_CG + n_columns_to_add ])
			n_columns_to_add += 1

		P_CG += n_columns_to_add

	### Add the columns with the highest negative reduced costs
		if n_columns_to_add>0:
			model = add_columns_Slope_SVM(X, y, model, violated_columns, violated_llambdas, slacks_vars, all_w_Slope)

			for column_to_add in violated_columns:
				idx_columns.append(column_to_add)
				columns_to_check.remove(column_to_add)

			print 'Number of columns added: '+str(n_columns_to_add)
			continue_loop = True


### Solution
	beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in idx_columns])
	beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in idx_columns])
	beta        = np.array(beta_plus) - np.array(beta_minus)
	print '\nLen support = '+str(len(np.where(beta!=0)[0]))

### Obj val
	obj_val = model.ObjVal
	print 'Obj value   = '+str(obj_val)

### Time stops
	time_ColG = time.time()-start_time
	print 'Time ColG = '+str(round(time_ColG, 3))

	return model


