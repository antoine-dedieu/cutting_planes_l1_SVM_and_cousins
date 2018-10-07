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
from L1_SVM_ConG import *



### Simulation parameters
type_Sigma = 2
N, P   = 10000, 100
k0     = 10
rho    = 0.1
mu     = 1
seed_X = random.randint(0,1000)


### Algorithm parameters
epsilon_RC  = 1e-2


### Simulate data		
X_train, l2_X_train, y_train = simulate_data(type_Sigma, N, P, k0, rho, mu, seed_X)


### Regularization coefficient
llambda_max = np.max(np.sum( np.abs(X_train), axis=0)) #infinite norm of the sum over the lines
llambda 	  = 1e-2 * llambda_max




#-----------------------------------------MODEL 1-----------------------------------------

print '\n\n###### L1 SVM: LP SOLVER ON ALL FEATURES #####'

idx_samples = range(N)
_ = L1_SVM_ConG(X_train, y_train, idx_samples, llambda) 



#-----------------------------------------MODEL 2-----------------------------------------

print '\n\n###### L1 SVM: SUBSAMPLING FIRST ORDER METHOD + CONSTRAINT GENERATION #####'
start_time = time.time()

index_samples_FOM = subsampling_FOM_n_large(X_train, y_train, llambda)
_ = L1_SVM_ConG(X_train, y_train, index_samples_FOM, llambda, epsilon_RC=epsilon_RC)

print '\n# Total time subsampling FOM + ConG = '+str( round(time.time()-start_time, 3) )


