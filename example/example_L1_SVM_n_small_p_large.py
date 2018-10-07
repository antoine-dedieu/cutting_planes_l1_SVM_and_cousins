import numpy as np
import datetime
import time
import os
import sys
import math
import random

from simulate_data import *

sys.path.append('../L1_Slope_SVM_algorithms')
from FOM_L1_Slope import *
from L1_SVM_ColG import *



### Simulation parameters
type_Sigma = 2
N, P   = 100, 10000
k0     = 10
rho    = 0.1
mu     = 1
seed_X = random.randint(0,1000)


### Algorithm parameter
epsilon_RC  = 1e-2

### Simulate data		
X_train, l2_X_train, y_train = simulate_data(type_Sigma, N, P, k0, rho, mu, seed_X)


### Regularization coefficient
llambda_max = np.max(np.sum( np.abs(X_train), axis=0)) #infinite norm of the sum over the lines
llambda 	  = 1e-2 * llambda_max



#-----------------------------------------MODEL 1-----------------------------------------

print '\n\n###### L1 SVM: LP SOLVER ON ALL FEATURES #####'

idx_columns = range(P)
_ = L1_SVM_ColG(X_train, y_train, idx_columns, llambda)




#-----------------------------------------MODEL 2-----------------------------------------

print '\n\n###### L1 SVM: COLUMN GENERATION WITH RANDOM INITIALIZATION #####'

n_columns = 50
idx_columns_random  = random.sample(range(P), n_columns)
_ = L1_SVM_ColG(X_train, y_train, idx_columns_random, llambda, epsilon_RC=epsilon_RC)




#-----------------------------------------MODEL 3-----------------------------------------

print '\n\n###### L1 SVM: COLUMN GENERATION WITH CORRELATION SCREENING #####'

idx_columns_correl = init_correlation(X_train, y_train, n_columns)
_ = L1_SVM_ColG(X_train, y_train, idx_columns_correl, llambda, epsilon_RC=epsilon_RC)




#-----------------------------------------MODEL 4-----------------------------------------

print '\n\n###### L1 SVM: COLUMN GENERATION VIA REGULARIZATION PATH ALGORITHM #####'

start_time = time.time()
llambda_bis        = .5*llambda_max #lambda decreased along the regularization path
idx_columns_correl = init_correlation(X_train, y_train, 10)
model_reg_path     = 0 #initialize model

while llambda_bis > llambda:
	model_reg_path = L1_SVM_ColG(X_train, y_train, idx_columns_correl, llambda_bis, epsilon_RC=epsilon_RC, model=model_reg_path)
	llambda_bis     *= 0.7

_ = L1_SVM_ColG(X_train, y_train, idx_columns_correl, llambda, epsilon_RC=epsilon_RC, model=model_reg_path)
print '\n# Total time regularization path = '+str( round(time.time()-start_time, 3) )




#-----------------------------------------MODEL 5 -----------------------------------------

print '\n\n###### L1 SVM: FIRST ORDER METHOD + COLUMN GENERATION #####'
start_time = time.time()

#Correlation screening
idx_columns_correl = init_correlation(X_train, y_train, 10*N)
X_train_reduced     = np.array([X_train[:,j] for j in idx_columns_correl]).T 

#FOM
idx_columns_FOM_restricted, _ = loop_FOM('L1', X_train_reduced, y_train, [llambda], tau_max=0.2, n_loop=1, T_max=200)
idx_columns_FOM 		      = np.array(idx_columns_correl)[idx_columns_FOM_restricted].tolist()

#ColG
_ = L1_SVM_ColG(X_train, y_train, idx_columns_FOM, llambda, epsilon_RC=epsilon_RC)
print '\n# Total time FOM + ColG = '+str( round(time.time()-start_time, 3) )
















