import numpy as np
import time
import math
import random

#-----------------------------------------------SUBSAMPLING HEURISTIC FOR FOM WHEN N IS SMALL--------------------------------------------------

def subsampling_FOM_n_large_p_large(X, y, llambda, tau_max=0.2, n_loop=5, T_max=100):

####### INPUT #######
# X, y     : design matrix and prediction vector
# llambda  : regularization parameter


####### OUTPUT #######
# idx_samples : list of samples to use to initialize constraint generation


	start_time = time.time()
	N, P = X.shape
	print '\n# 1/ Run First Order Method'

### Parameters
	(N0, P0) = (5*int(math.sqrt(N)), 5*int(math.sqrt(P)))

### Averaged estimator
	old_beta_averaged = -np.ones(P+1)
	beta_averaged     = np.zeros(P+1)

### Convergence criterion
	CV_criterion = 1e6 
	n_sample     = 0
	while CV_criterion>1e-1 and n_sample<int(N/N0):
		n_sample += 1
		print 'Sample number '+str(n_sample)
		
	### Subsample from rows of X and from y
		subset 	    = random.sample(xrange(N),N0)
		X_subsample = X[subset,:] 
		y_subsample = y[subset] 

	### Correlation threshold
		argsort_columns         = np.argsort(np.abs(np.dot(X_subsample.T, y_subsample) ))
		argsort_columns_reduced = argsort_columns[::-1][:P0]
		X_subsample_reduced     = np.array([X_subsample[:,j] for j in argsort_columns_reduced]).T

	### Adapt regularization coefficient
		llambda_subsample = llambda*N0/N

	### Run FOM with columns restriction
		beta_sample_reduced, idx_columns_reduced = loop_FOM_with_columns_restriction(X_subsample_reduced, y_subsample, llambda_subsample, tau_max=tau_max, n_loop=n_loop, T_max=T_max, N_original=N)

	### Estimator on all the columns
		beta_sample = np.zeros(P+1)
		for i in range(len(idx_columns_reduced)-1): beta_sample[ argsort_columns_reduced[ idx_columns_reduced[i] ] ] = beta_sample_reduced[i]
		beta_sample[-1] = beta_sample_reduced[-1]

	### Convergence test
		old_beta_averaged = np.copy(beta_averaged)
		beta_averaged    += np.array(beta_sample)
		CV_criterion      = np.linalg.norm(1./max(1,n_sample)*beta_averaged - 1./max(1,n_sample-1)*old_beta_averaged)


### Averaged estimator
	beta_averaged *= 1./n_sample
	ones_N 		   = np.ones(N)

### Hard thresholding on the averaged estimator
	idx_columns = np.argsort(np.abs(beta_averaged[:-1]))[::-1][:200]
	print 'Len support: '+str(len(idx_columns))

### Determine the set of constraints violated by the averaged estimator
	constraints    = ones_N - y*( np.dot(X, beta_averaged[:-1]) + beta_averaged[-1]*ones_N)
	idx_samples    = np.arange(N)[constraints >= 0]
	print 'Number violated constraints: '+str(idx_samples.shape[0])

### Time 
	time_smoothing = time.time()-start_time
	print 'Time FOM: '+str(round(time_smoothing, 3))

	return idx_samples.tolist(), idx_columns.tolist()



#-----------------------------------------------RUN NESTEROV'S WITH CONTINUATION OVER TAU AND COLUMNS RESTRICTION--------------------------------------------------

# After the first iteration over tau, we restrict X, and y to the 'relevant' columns
# We do not need this method for Slope

def loop_FOM_with_columns_restriction(X, y, llambda, tau_max=0.2, n_loop=1, T_max=200, N_original=0):
	
####### INPUT #######

# X, y        : design matrix and prediction vector - in pratice X corresponds to X_reduced in subsampling_FOM_n_large_p_large
# llambda     : regularization parameter
# idx_start   : features used
# tau_max     : smoothing value to start with
# n_loop.     : number of loops (over tau)
# T_max       : maximum number of iterations for loop


####### OUTPUT #######

# beta_FOM    : estimator restricted to idx_columns
# idx_columns : sublist of columns of X 
	

	start_time = time.time()
	N, P       = X.shape
	beta_FOM   = np.zeros(P+1)
	ones_N     = np.ones(N)


### Design matrix with additional column (for intercept term)
	X_add       = 1/math.sqrt(N_original)*np.ones((N, P+1))
	X_add[:,:P] = X

### Prepare for restriction
	X_reduced  = np.copy(X)
	y_reduced  = np.copy(y)

### Highest eigenvalue of X_add
	highest_eig = power_method(X_add)

	tau = tau_max
	for loop in range(n_loop):
		beta_FOM = FOM_algo('L1', X_reduced, y_reduced, [llambda], beta_FOM, X_add, highest_eig, tau, T_max)		

	## After first loop, we restrict X and y to 'relevant' samples
		if loop == 0:
		### Only keep 100 highest columns
			idx_columns = np.argsort(np.abs(np.dot(X_reduced.T, y_reduced) ))[::-1][:100]
			P_reduced   = len(idx_columns)
			X_reduced   = X_reduced[:, idx_columns] 

			beta_FOM    = beta_FOM[idx_columns.tolist() + [P_reduced]]

		### Restrict X_add and highest_eig accordingly
			X_add         		 = 1/math.sqrt(N_original)*np.ones((N, P_reduced+1))
			X_add[:,:P_reduced]  = X_reduced
			highest_eig  		 = power_method(X_add)

		tau   = 0.7*tau
		T_max = 20 #limit number of iterations after first run

	return beta_FOM, idx_columns.tolist()







#-----------------------------------------------SUBSAMPLING HEURISTIC FOR FOM WHEN N IS SMALL--------------------------------------------------

def subsampling_FOM_n_large(X, y, llambda, tau_max=0.1, n_loop=5, T_max=20):

####### INPUT #######
# X, y     : design matrix and prediction vector
# llambda  : regularization parameter


####### OUTPUT #######
# idx_samples : list of samples to use to initialize constraint generation


	start_time = time.time()
	N, P = X.shape
	print '\n# 1/ Run First Order Method'

### Parameter
	N0 = 10*P

### Averaged estimator
	old_beta_averaged = -np.ones(P+1)
	beta_averaged     = np.zeros(P+1)


### Convergence criterion
	CV_criterion = 1e6 
	n_sample     = 0
	while CV_criterion>1e-1 and n_sample<int(N/N0):
		n_sample += 1
		print 'Sample number '+str(n_sample)
		
	### Subsample from rows of X and y
		subset 	    = random.sample(xrange(N),N0)
		X_subsample = X[subset,:] 
		y_subsample = y[subset] 

	### Adapt regularization coefficient
		llambda_subsample = llambda*N0/N

	### Run FOM with columns restriction
		beta_sample = loop_FOM_with_samples_restriction(X_subsample, y_subsample, llambda_subsample, tau_max=tau_max, n_loop=n_loop, T_max=T_max, N_original=N)

		old_beta_averaged = np.copy(beta_averaged)
		beta_averaged    += np.array(beta_sample)
		CV_criterion      = np.linalg.norm(1./max(1,n_sample)*beta_averaged - 1./max(1,n_sample-1)*old_beta_averaged)

### Determine the set of violated constraints by the averaged estimator
	beta_averaged *= 1./n_sample
	ones_N 		   = np.ones(N)

	constraints    = ones_N - y*( np.dot(X, beta_averaged[:-1]) + beta_averaged[-1]*ones_N)
	idx_samples    = np.arange(N)[constraints >= 0]
	print 'Number violated constraints: '+str(idx_samples.shape[0])

### Time 
	time_smoothing = time.time()-start_time
	print 'Time FOM: '+str(round(time_smoothing, 3))

	return idx_samples.tolist()






#-----------------------------------------------RUN NESTEROV'S WITH CONTINUATION OVER TAU AND CONSTRAINT RESTRICTION--------------------------------------------------

# After the first iteration over tau, we restrict X, and y to the 'relevant' samples
# We do not need this method for Slope

def loop_FOM_with_samples_restriction(X, y, llambda, tau_max=0.2, n_loop=1, T_max=200, N_original=0):
	
####### INPUT #######

# X, y        : design matrix and prediction vector
# llambda       : regularization parameter
# idx_start   : features used
# tau_max     : smoothing value to start with
# n_loop.     : number of loops (over tau)
# T_max       : maximum number of iterations for loop


####### OUTPUT #######

# beta_FOM : FOM estimator
	

	start_time = time.time()
	N, P       = X.shape
	beta_FOM   = np.zeros(P+1)
	ones_N     = np.ones(N)


### Design matrix with additional column (for intercept term)
	X_add       = 1/math.sqrt(N_original)*np.ones((N, P+1))
	X_add[:,:P] = X

### Prepare for restriction
	X_reduced  = np.copy(X)
	y_reduced  = np.copy(y)

### Highest eigenvalue of X_add
	highest_eig = power_method(X_add)

	tau = tau_max
	for loop in range(n_loop):
		beta_FOM = FOM_algo('L1', X_reduced, y_reduced, [llambda], beta_FOM, X_add, highest_eig, tau, T_max)

	## After first loop, we restrict X and y to 'relevant' samples
		if loop == 0:
			constraints = ones_N - y*( np.dot(X, beta_FOM[:P]) + beta_FOM[P]*ones_N) 

		### Relevant samples are defined at at leat .05 from the margin
			idx_samples = np.arange(N)[constraints >= -.05]

		### Restrict X and y
			X_reduced = X_reduced[idx_samples,:] 
			y_reduced = np.array(y_reduced)[idx_samples]
			N_reduced = len(idx_samples)

		### Restrict X_add and highest_eig accordingly
			X_add       = 1/math.sqrt(N)*np.ones((N_reduced, P+1))
			X_add[:,:P] = X_reduced
			highest_eig = power_method(X_add)

		tau = 0.7*tau

	return beta_FOM





#-----------------------------------------------RUN FIRST ORDER WITH CONTINUATION OVER TAU--------------------------------------------------

# We run this method for L1 and Slope

def loop_FOM(type_reg, X, y, llambda_arr, tau_max=0.2, n_loop=1, T_max=200):
	
####### INPUT #######

# type_reg.   : 'L1' or 'Slope'
# X, y        : design matrix and prediction vector
# llambda_arr  : regularization parameter if L1 / sequence of Slope parameters if Slope
# tau_max     : smoothing value to start with
# n_loop.     : number of loops (over tau)
# T_max       : maximum number of iterations for loop


####### OUTPUT #######

# idx_columns : columns returned by the first order method
# beta_FOM_restricted : FOM estimator restricted to its support
	

	start_time = time.time()
	print '\n# 1/ Run First Order Method'

	N, P       = X.shape
	beta_FOM   = np.zeros(P+1)

### Design matrix with additional column (for intercept term)
	X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
	X_add[:,:P] = X

### Highest eigenvalue of X_add
	highest_eig = power_method(X_add)


	tau = tau_max
	for _ in range(n_loop):
		beta_FOM = FOM_algo(type_reg, X, y, llambda_arr, beta_FOM, X_add, highest_eig, tau, T_max)
		tau = 0.7*tau


### Support
	idx_columns = np.where(beta_FOM[:P] !=0)[0]
	beta_FOM_restricted	= beta_FOM[idx_columns]
	print 'Len support smoothing: '+str(idx_columns.shape[0])

### Time 
	time_smoothing = time.time()-start_time
	print 'Time FOM: '+str(round(time_smoothing, 3))

	return idx_columns, beta_FOM_restricted





#-----------------------------------------------FIRST ORDER ALGORITHM--------------------------------------------------

def FOM_algo(type_reg, X, y, llambda_arr, beta_start, X_add, highest_eig, tau, T_max):

####### INPUT #######

# type_reg    : 'L1' or 'Slope'
# X, y        : design matrix and prediction vector
# llambda_arr  : regularization parameter if L1 / sequence of Slope parameters if Slope
# beta_start  : vector to start with
# X_add       : X matrix with additional columns of 1/sqrt(n) for intercept coefficients
# highest_eig : highest eigenvaue computed via power method of X_add^T * X_add
# tau         : smooyhing parameter
# T_max       : maximum number of iterations for loop


####### OUTPUT #######

# beta_m     : FOM estimator 

	N, P  = X.shape

### Parameters  
	old_beta = np.ones(P+1)
	beta_m   = beta_start
	ones_N    = np.ones(N)
	
### Parameters for AGD
	t_AGD_old 	   = 1
	t_AGD          = 1
	eta_m_old 	   = beta_start
	Lipchtiz_coeff = highest_eig/(4*tau) 

	
	loop=0
	while (np.linalg.norm(beta_m-old_beta)>1e-3 and loop<T_max): 
		loop+=1
		
	### Compute the gradient og the smooth hinge loss
		aux 	      = ones_N - y*np.dot(X_add,beta_m)
		w_tau         = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
		
		gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])
		gradient_loss = -0.5*np.dot(X_add.T, gradient_aux) 

	### Gradient descent step
		old_beta = beta_m 
		grad     = beta_m - 1/float(Lipchtiz_coeff)*gradient_loss

	### Thresholding
		if type_reg=='L1':
			eta_m = np.array([ soft_thresholding(grad[i], llambda_arr[0]/Lipchtiz_coeff) for i in range(P)] + [grad[P]])

		elif type_reg=='Slope':
			eta_m = np.array( thresholding_slope(grad[:P], llambda_arr/Lipchtiz_coeff).tolist() + [grad[P]])
   
	### Run AGD
		t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
		aux_t_AGD = (t_AGD_old-1)/t_AGD

		beta_m     = eta_m + aux_t_AGD * (eta_m - eta_m_old)

		t_AGD_old = t_AGD
		eta_m_old = eta_m
	
	beta_m[P] /= math.sqrt(N)
	return beta_m



	

#---------------------------------------------------------CORRELATION SCREENING---------------------------------------------------------

def init_correlation(X, y, n_columns):

####### INPUT #######
# X, y        : design matrix and prediction vector
# n_columns.  : number of columns to keep after correlation screeening

####### OUTPUT #######
# idx_columns_correl: list of columns

	start_correl       = time.time()
	argsort_columns    = np.argsort(np.abs( np.dot(X.T, y) ))
	idx_columns_correl = argsort_columns[::-1][:n_columns]
	print 'Time correlation screening = '+str( round(time.time() - start_correl, 3) )
	return idx_columns_correl.tolist()




#-----------------------------------------------POWER METHOD TO COMPUTE HIGHEST EIGENVALUE of XTX--------------------------------------------------

def power_method(X):

####### INPUT #######
# X: matrix

####### OUTPUT #######
# highest_eig: highest eigenvalue of XTX

	P = X.shape[1]

	highest_eigvctr     = np.random.rand(P)
	old_highest_eigvctr = -1
	
	while(np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2):
		old_highest_eigvctr = highest_eigvctr
		highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))
		highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
	
	X_highest_eig = np.dot(X, highest_eigvctr)

	highest_eig   = np.dot(X_highest_eig.T, X_highest_eig)/np.linalg.norm(highest_eigvctr)
	return highest_eig



#-----------------------------------------------THRESHOLDING OPERATOR FOR L1 NORM--------------------------------------------------

# Please refer to paper

def soft_thresholding(c,llambda):

	if(llambda>=abs(c)):
		return 0
	else:
		if (c>=0):
			return c-llambda
		else:
			return c+llambda
	



#-----------------------------------------------THRESHOLDING OPERATOR FOR SLOPE NORM--------------------------------------------------


def thresholding_slope(c_arr, llambda_arr):

####### INPUT #######
# c_arr      : vector
# llambda_arr: Slope regularization=non negative and non increasing sequence of penalizations

####### OUTPUT #######
# beta_slope : solution of Slope thresholding problem

	sign  = np.sign(c_arr)
	y_abs = np.abs(c_arr)

	arg    = np.argsort(y_abs)
	result = thresholding_slope_positive(y_abs[arg][::-1], llambda_arr)[::-1] #sorted

	arg_bis    = np.argsort(arg)
	beta_slope = result[arg_bis]*sign

	return beta_slope



def thresholding_slope_positive(c_arr, llambda_arr):

####### INPUT #######
#c_arr		: non negative and non increasing sequence of float
#llambda_arr: non negative and non increasing sequence of penalizations

####### OUTPUT #######
# result : solution of Slope thresholding problem


	n      = len(c_arr)
	idx_i  = np.zeros(n)
	idx_j  = np.zeros(n)
	s      = np.zeros(n)
	w      = np.zeros(n)
	result = np.zeros(n)

	k = 0;
	for i in range(n):
		idx_i[k] = i
		idx_j[k] = i
		s[k]     = c_arr[i] - llambda_arr[i]
		w[k]     = max(0, s[k])
		 
		while k>0 and w[k-1] <= w[k]:
			k -= 1
			idx_j[k] = i
			s[k]    += s[k+1]
			w[k]     = max(0, s[k] / (i - idx_i[k] + 1))
		k += 1
		#print idx_i, idx_j, w
	
	for j in range(k):
		for i in range(int(idx_i[j]), int(idx_j[j]+1)):
			result[i] = w[j]

	return result



