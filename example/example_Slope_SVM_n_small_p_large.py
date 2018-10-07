import numpy as np
import datetime
import time
import os
import sys
import math
import random
import cvxpy as cvx

from simulate_data_group import *

sys.path.append('../L1_Slope_SVM_algorithms')
from FOM_L1_Slope import *
from Slope_SVM_ColG import *



### Simulation parameters
type_Sigma = 2

N, P   = 100, 10000
k0     = 10
rho    = 0.1
mu     = 1
seed_X = random.randint(0,1000)


### Algorithm parameters
epsilon_RC = 1e-2
TYPE_SIMU  = 1 #{1,2}: 1 compares our method against CVX on simple regularization / 2: only runs our model


### Simulate data		
X_train, l2_X_train, y_train = simulate_data(type_Sigma, N, P, k0, rho, mu, seed_X)


### Regularization coefficient
llambda_max  = np.max(np.sum( np.abs(X_train), axis=0)) #infinite norm of the sum over the lines
llambda 	   = 1e-2 * llambda_max


if TYPE_SIMU==1:   llambda_arr = np.array([2*llambda for _ in range(k0)] + [llambda for _ in range(P-k0)])

elif TYPE_SIMU==2: llambda_arr = np.array([llambda * math.sqrt(math.log(2.*P/(j+1))) for j in range(P)])



#-----------------------------------------MODEL 1-----------------------------------------

print '\n\n###### SLOPE SVM: FIRST ORDER METHOD + COLUMN GENERATION #####'
start_time = time.time()

#Correlation screening
idx_columns_correl = init_correlation(X_train, y_train, 10*N)
X_train_reduced     = np.array([X_train[:,j] for j in idx_columns_correl]).T 

#FOM
idx_columns_FOM_restricted, beta_FOM_restricted = loop_FOM('Slope', X_train_reduced, y_train, llambda_arr, tau_max=0.2, T_max=200)
idx_columns_FOM = np.array(idx_columns_correl)[idx_columns_FOM_restricted].tolist()

#ColG
_ = Slope_SVM_ColG(X_train, y_train, idx_columns_FOM, llambda_arr, beta_FOM_restricted, epsilon_RC=epsilon_RC)
print '\n# Total time FOM + ColG = '+str( round(time.time()-start_time, 3) )





#-----------------------------------------MODEL 2: CVXPY-----------------------------------------

print '\n\n###### SLOPE SVM: CVXPY #####'

def CVX_Slope(X_train, y_train, llambda, k_Slope, solver):
	start_time = time.time()
	N, P = X_train.shape

	beta  = cvx.Variable(P)
	beta0 = cvx.Variable()
	loss  = cvx.sum(cvx.pos(1 - cvx.multiply(y_train, X_train*beta + beta0)))
	reg1  = cvx.norm(beta, 1)

	#for j in range(min(P-1, K_max)): reg2  += (lambda_arr[j]-lambda_arr[j+1]) * sum_largest(abs(beta), j+1)  
	reg2 = cvx.sum_largest(cvx.abs(beta), k_Slope) 

	prob  = cvx.Problem(cvx.Minimize(loss + llambda*reg1 + llambda*reg2))
	dict_solver = {'gurobi':cvx.GUROBI, 'ecos':cvx.ECOS}

	prob.solve(solver=dict_solver[solver])

### Solution
	support = np.where(np.round(beta.value,6)!=0)[0]
	print '\nLen support '+solver+' = '+str(len(support))

### Obj val
	obj_val = prob.value
	print 'Obj value '+solver+'  = '+str(obj_val)

### Time stops
	time_cvx = time.time()-start_time
	print 'Time CVX '+solver+' = '+str(round(time_cvx,2))



if TYPE_SIMU==1: 
	print '\n# SOLVER GUROBI'
	CVX_Slope(X_train, y_train, llambda, k0, 'gurobi')

	print '\n# SOLVER ECOS'
	CVX_Slope(X_train, y_train, llambda, k0, 'ecos')
				



