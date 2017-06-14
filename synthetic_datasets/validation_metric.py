import numpy as np



def validation_metric(betas, train_errors, X_pop_val, y_pop_val, beta_min_pop_val, l2_X_train, name_metric, K0_list, N_alpha):

# X_pop_val : non standardized
# l2_X_train: for beats standardization


	argmin_betas   = [] #4 betas minimizing the metric
	best_k_plot    = [] #list of K0 associated to 4 best betas
	all_k_plot     = []
	

	#l2_estimation
	beta_min_pop_norm = beta_min_pop_val/np.linalg.norm(beta_min_pop_val)

	#misclassification
	N_val      = X_pop_val.shape[0]
	ones_N_val = np.ones(N_val)



#---Grid of L1+L0 and L2+L0
	for aux in range(2):
		betas_aux = betas[aux]
		errors    = train_errors[aux]


	#---l2_estimation
		if name_metric == 'l2_estimation':
			metric_list  = [[0 for loop in range(N_alpha) ] for K0 in K0_list]
			for K0 in K0_list:
				for loop in range(N_alpha):
					beta                  = betas_aux[K0][loop][0]
					metric_list[K0][loop] = np.linalg.norm(beta/(1e-10+np.linalg.norm(beta)) - beta_min_pop_norm)


	#---misclassification
		if name_metric == 'misclassification':
			metric_list = [[] for K0 in K0_list]
			for K0 in K0_list:
				metric_list[K0] = y_pop_val*np.sign([1e-10 + np.dot(X_pop_val, l2_X_train*beta[0]) + beta[1]*ones_N_val for beta in betas_aux[K0]])
				metric_list[K0] = [np.sum(misc<0)/float(N_val) for misc in metric_list[K0]]
		

		argmin_list = [np.argmin(metric_list[K0]) for K0 in K0_list]
		argmin_K0   = np.argmin([metric_list[K0][argmin_list[K0]] for K0 in K0_list])
		

		argmin_betas.append(betas_aux[argmin_K0][argmin_list[argmin_K0]])
		best_k_plot.append(( errors[argmin_K0], metric_list[argmin_K0], argmin_K0))
		all_k_plot.append(metric_list)



#---L1 and L2
	for aux in range(2,4):
		betas_aux= betas[aux]
		errors   = train_errors[aux]
		
		if name_metric == 'l2_estimation':
			metric_list  = [0 for loop in range(N_alpha) ]
			for loop in range(N_alpha):
				beta              = betas_aux[loop][0]
				metric_list[loop] = np.linalg.norm(beta/(1e-10+np.linalg.norm(beta)) - beta_min_pop_norm)

		if name_metric == 'misclassification':
			metric_list = y_pop_val*np.sign([1e-10 + np.dot(X_pop_val, l2_X_train*beta[0]) + beta[1]*ones_N_val for beta in betas_aux])
			metric_list = [np.sum(misc<0)/float(N_val) for misc in metric_list]


		argmin_betas.append(betas_aux[np.argmin(metric_list)])
		best_k_plot.append(( errors, metric_list, 0))
		all_k_plot.append(metric_list)


	return argmin_betas, best_k_plot, all_k_plot


