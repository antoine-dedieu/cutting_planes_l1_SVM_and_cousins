import numpy as np
import time
import math

import sys
sys.path.append('../L1_Slope_SVM_algorithms')
from FOM_L1_Slope import *

sys.path.append('spgl1') #cloned from https://github.com/drrelyea/SPGL1_python_port
from spgl1 import spg_lasso 


#-----------------------------------------------FIRST ORDER ALGORITHM FOR GROUP SVM--------------------------------------------------

def FOM_group(type_algo, X, y, group_to_feat, llambda, tau_max=0.2, n_loop=1, T_max=200):

####### INPUT #######

# type_algo     : algorithm used for first order method {'AGD', 'block_CD'}
# X, y          : design matrix and prediction vector
# group_to_feat : for each index of a group we have access to the list of features belonging to this group
# llambda       : regularization parameter
# n_loop.       : number of loops (over tau)
# T_max         : maximum number of iterations for loop


####### OUTPUT #######

# idx_groups   : indexes of groups to be included
	

	start_time     = time.time()
	print '\n# 1/ Run First Order Method'

	N, P           = X.shape
	beta_FOM_group = np.zeros(P+1)
	ones_N    	   = np.ones(N)

### Design matrix with additional column (for intercept term)
	X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
	X_add[:,:P] = X
	tau = tau_max


### List of highest eigenvalues + values of regularization parameter used for FORM

### CASE 1: block CD, we need each eigenvalue on each submatrix + a decreasing sequence of llambda values
	if type_algo == 'block_CD':
		n_groups         = len(group_to_feat)
		highest_eig_list = np.zeros(n_groups)

		for i in range(n_groups):
			len_group               = len(group_to_feat[i])
			X_add_bis               = 1/math.sqrt(N)*np.ones((N, len_group+1))
			X_add_bis[:,:len_group] = X[:,group_to_feat[i]]
			highest_eig_list[i]     = power_method(X_add)

		#Run block CD for decreasing sequence of llambda
		llambda_list = [5*llambda, 2*llambda, llambda]

### CASE 2: AGD, we just need one eigenvalue of X_add^T * X_add
	elif type_algo == 'AGD':
		highest_eig_list = np.array([power_method(X_add)])
		llambda_list 	 = [llambda]


### If block CD, we loop over the list of llambda values

	for llambda in llambda_list: 
		tau = tau_max

		for loop in range(n_loop):
			beta_FOM_group, active_loop = FOM_group_algo(type_algo, X, y, group_to_feat, llambda, beta_FOM_group, X_add, highest_eig_list, tau, T_max)
			tau = 0.7*tau

### Active groups
	idx_groups_FOM = np.where(active_loop==True)[0]
	print 'Number active groups: '+str(len(idx_groups_FOM))

### Time 
	time_smoothing = time.time()-start_time
	print 'Time FOM: '+str(round(time_smoothing, 3))

	return idx_groups_FOM







#-------------------------------------------------ALGORITHM----------------------------------------------------

def FOM_group_algo(type_algo, X, y, group_to_feat, llambda, beta_start, X_add, highest_eig_list, tau, T_max):
	
####### INPUT #######

# type_algo       : algorithm used for first order method {'AGD', 'block_CD'}
# X, y            : design matrix and prediction vector
# group_to_feat   : for each index of a group we have access to the list of features belonging to this group
# llambda           : regularization parameter
# beta_start      : vector to start with
# X_add           : X matrix with additional columns of 1/sqrt(n) for intercept coefficients
# highest_eig_list: list of highest eigenvaue computed via power method of
					  # - X_add^T * X_add if AGD used
					  # - all submatrices if block CD used
# tau             : smoothing parameter
# T_max           : maximum number of iterations for loop


####### OUTPUT #######
	
# beta_m      : FOM estimator

	
	N, P  = X.shape
	G     = len(group_to_feat) #number of groups

### Parameters  
	old_beta = np.ones(P+1)
	beta_m   = beta_start
	ones_N    = np.ones(N)

	Lipchtiz_coeff  = highest_eig_list/(4*tau) 


### Parameters for AGD
	t_AGD_old      = 1
	t_AGD          = 1
	eta_m_old      = beta_start


### Parameters for block CD
	X_beta_group = np.dot(X_add, beta_start) #fast residual updates
	active_loop  = np.array([True for _ in range(G)]) #active set strategy
	
	len_supp_act_loop     = 0
	old_len_supp_act_loop = -1 

	
	loop=0
	while(np.linalg.norm(beta_m-old_beta)>1e-3 and loop<T_max and len_supp_act_loop != old_len_supp_act_loop ): #stop block CD if active set stabilizes
		loop += 1
		old_beta = np.copy(beta_m) 


	### If BLOCK CD, loop over the different active groups If AGD, loop f sie 1
		for idx in range(Lipchtiz_coeff.shape[0]):

		### If block CD, restrict to active groups
			if active_loop[idx] == True:

			### Compute 1-y*XT*beta
				dict_X_beta = {'block_CD':X_beta_group, 'AGD':np.dot(X_add, beta_m)}
				group_idx   = list(group_to_feat[idx])+[P]
				
			### Compute gradient of smotth hinge loss
				aux 		  = ones_N - y*dict_X_beta[type_algo]
				w_tau         = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
				gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])


			### Gradient on the group if block CD, full gradient if AGD
				dict_gradient = {'block_CD':-0.5*np.dot(X_add[:, group_idx].T, gradient_aux), 
								 'AGD'     :-0.5*np.dot(X_add.T, gradient_aux)}
				
				gradient_loss = dict_gradient[type_algo]



			### Thresholding operators + Gradient step 

			### CASE 1: block CD
				if type_algo == 'block_CD':

				### L_inf thresholding on the group
					old_beta_m_group  = np.copy(beta_m[group_idx])
					grad              = beta_m[group_idx] - 1./Lipchtiz_coeff[idx] * gradient_loss
					soft_thresholding = thresholding_linf(grad[:-1], llambda/Lipchtiz_coeff[idx]) 

				### If group set to zero, delete it from active set
					block_not_null = not np.all([soft_thresholding_coef == 0 for soft_thresholding_coef in soft_thresholding])

					active_loop[idx]           = block_not_null
					beta_m[group_to_feat[idx]] = soft_thresholding
					beta_m[P]                  = grad[-1]

				### Residual updates of X*beta
					delta_beta_group   = np.dot(X_add[:, group_idx], beta_m[group_idx]-old_beta_m_group)
					X_beta_group      += delta_beta_group
				


			### CASE 2: AGD 
				elif type_algo == 'AGD':
					st = time.time()

				### Gradien step
					grad = beta_m - 1/float(np.array(Lipchtiz_coeff)[idx])*gradient_loss   #Not block CD
					
				### Thresholding step
					eta_m = np.array(thresholding_l1_linf(grad[:P], llambda/Lipchtiz_coeff[idx], group_to_feat) + [grad[P]]) #idx = 0
				
				### AGD 
					t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
					aux_t_AGD = (t_AGD_old-1)/t_AGD

					beta_m     = eta_m + aux_t_AGD * (eta_m - eta_m_old)

					t_AGD_old = t_AGD
					eta_m_old = eta_m

					active_loop = np.array([not np.all([beta_m[i] == 0 for i in group_to_feat[j]]) for j in range(G)])
					

	### Test whether same sparsity
		old_len_supp_act_loop = min(len_supp_act_loop, G/2+.5) #we dont want to stop during first iterations
		len_supp_act_loop     = len(np.where(active_loop==True)[0])

	return beta_m, active_loop



#------------------------------------------------------CORRELATION SCREENING ON GROUPS---------------------------------------------------------

def init_group_correlation(X, y, group_to_feat, n_groups):

####### INPUT #######
# X, y          : design matrix and prediction vector
# group_to_feat : for each index of a group we have access to the list of features belonging to this group
# n_groups      : number of groups to keep after correlation screeening

####### OUTPUT #######
# idx_groups_correl: list of groups

	start_correl     = time.time()
	abs_correlations      = np.abs(np.dot(X.T, y))       #class are balanced so we always can compute this way

	#Sum of correl in groups
	sum_abs_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat] 
	argsort_columns       = np.argsort(sum_abs_correl_groups)
	idx_groups_correl 	  = argsort_columns[::-1][:n_groups]
	print 'Time correlation screening = '+str( round(time.time() - start_correl, 3) )
	
	return list(idx_groups_correl)



#-----------------------------------------------THRESHOLDING OPERATOR FOR L_INF NORM--------------------------------------------------

### We use Moreau identity to compute thresholding for l_inf norm

def thresholding_linf(c_group, llambda):

####### INPUT #######
# c_group : vector of coefficients on groups
# llambda : regularization parameter

####### OUTPUT #######
# beta_slope : solution of l_inf thresholding problem

	if np.sum(np.abs(c_group)) > llambda:
		proj_l1_ball, _, _, _  = spg_lasso(np.identity(c_group.shape[0]), c_group, llambda) #computes projection onto l1 ball
	else:
		proj_l1_ball = c_group
	return c_group - proj_l1_ball


#-----------------------------------------------THRESHOLDING OPERATOR FOR L1-L_INF NORM--------------------------------------------------

### We loop over the groups

def thresholding_l1_linf(c, llambda, group_to_feat):

####### INPUT #######
# c       : list of vectors of coefficients on all groups
# llambda : regularization parameter

####### OUTPUT #######
# beta_slope : solution of l1_linf thresholding problem

	result = np.zeros(c.shape[0])

	for i in range(len(group_to_feat)):
		c_group       = c[group_to_feat[i]]
		l1_norm_group = np.sum(np.abs(c_group))
		
		if l1_norm_group > llambda:
			proj_l1_ball_group, _, _, _   = spg_lasso(np.identity(c_group.shape[0]), c_group, llambda)
			result[group_to_feat[i]]      = c_group - proj_l1_ball_group
	
	return list(result)




