import numpy as np
import datetime
import time
import os
import sys
import math
import random

from simulate_data_group import *

sys.path.append('../group_SVM_algorithms')
from FOM_group import *
from group_Linf_SVM_ColG import *


### Simulation parameters
N, P   = 100, 10000
k0     = 10
rho    = 0.1
mu     = 1
seed_X = random.randint(0,1000)
feat_by_group = 10
group_to_feat = np.array([range(10*k, 10*(k+1)) for k in range(P/feat_by_group)]) 


### Algorithm parameters
epsilon_RC     = 1e-2


### Simulate data		
X_train, l2_X_train, y_train = simulate_data_group(N, P, group_to_feat, k0, rho, mu, seed_X)


### Regularization parameter
abs_sum_cols = np.sum( np.abs(X_train), axis=0)
llambda_max  = np.max([np.sum(abs_sum_cols[idx]) for idx in group_to_feat]) #infinite norm of the sum over the groups
llambda 	 = 1e-1 * llambda_max



#-----------------------------------------MODEL 1-----------------------------------------

print '\n\n###### GROUP SVM: LP SOLVER ON ALL GROUPS #####'

idx_groups = range(group_to_feat.shape[0])
group_Linf_SVM_ColG(X_train, y_train, group_to_feat, idx_groups, llambda)



#-----------------------------------------MODEL 2-----------------------------------------

print '\n\n###### GROUP SVM: COLUMN GENERATION VIA REGULARIZATION PATH #####' 

start_time  = time.time()
llambda_bis = .5*llambda_max
n_groups    = 50 #number of groups to start with
idx_groups_correl = init_group_correlation(X_train, y_train, group_to_feat, n_groups)
model_reg_path    = 0 #model updated via regularization path

while llambda_bis > llambda:
	model_reg_path = group_Linf_SVM_ColG(X_train, y_train, group_to_feat, idx_groups_correl, llambda_bis, epsilon_RC=epsilon_RC, model=model_reg_path)
	llambda_bis     *= 0.7

_ = group_Linf_SVM_ColG(X_train, y_train, group_to_feat, idx_groups_correl, llambda, epsilon_RC=epsilon_RC, model=model_reg_path)
print '\n# Total time regularization path = '+str( round(time.time()-start_time, 3) )





#-----------------------------------------MODEL 3-----------------------------------------

### Correlation screening for models 3 and 4
start_time = time.time()
n_groups   = N #number of groups to start with
idx_groups_correl  = init_group_correlation(X_train, y_train, group_to_feat, N)

### Define X_train_reduced and group_to_feat_reduced accordingly to the groups selected
X_train_reduced       = np.zeros((N,0))
group_to_feat_reduced = []
aux = 0

for i in range(n_groups):
	X_train_reduced = np.concatenate([X_train_reduced, np.array([X_train[:,j] for j in group_to_feat[idx_groups_correl[i]]]).T ], axis=1)
	group_to_feat_reduced.append(range(aux, aux+len(group_to_feat[idx_groups_correl[i]]) ))
	aux += len(group_to_feat[idx_groups_correl[i]]) 
mid_time = time.time()-start_time




print '\n\n###### GROUP SVM: FOM (with AGD) + COLUMN GENERATION #####'

#FOM with AGD
idx_groups_reduced_FOM_AGD = FOM_group('AGD', X_train_reduced, y_train, group_to_feat_reduced, llambda, tau_max=0.2, T_max=200)
idx_groups_FOM_AGD 		   = np.array(idx_groups_correl)[idx_groups_reduced_FOM_AGD].tolist()

#ColG
_ = group_Linf_SVM_ColG(X_train, y_train, group_to_feat, idx_groups_FOM_AGD, llambda, epsilon_RC=epsilon_RC)
print '\n# Total time FOM (AGD) + ColG = '+str( round(time.time()-start_time, 3) )




#-----------------------------------------MODEL 4-----------------------------------------

print '\n\n###### GROUP SVM: FOM (with BLOCK CD) + COLUMN GENERATION  #####'
start_time = time.time()

#FOM with BLOCK CD
idx_groups_FOM_reduced_block_CD = FOM_group('block_CD', X_train_reduced, y_train, group_to_feat_reduced, llambda, tau_max=0.2, T_max=200)
idx_groups_FOM_block_CD 		= np.array(idx_groups_correl)[idx_groups_FOM_reduced_block_CD].tolist()

#ColG
_ = group_Linf_SVM_ColG(X_train, y_train, group_to_feat, idx_groups_FOM_block_CD, llambda, epsilon_RC=epsilon_RC)

print '\n# Total time FOM (BLOCK CD) + ColG = '+str( round(mid_time + time.time()-start_time, 3) )









