import numpy as np
import sys

sys.path.append('../graphics')
from bars_metrics import *


def plot_test_metrics(argmin_betas, X_pop_test, y_pop_test, beta_min_pop_test, l2_X_train, u_positive, pathname, name_metric):
	
#---Metrics
	#L2 estimation
	beta_min_pop_norm_test = beta_min_pop_test/np.linalg.norm(beta_min_pop_test)

	l2_estimation = []
	for i in range(4): 
	    beta = argmin_betas[i][0]
	    l2_estimation.append(np.linalg.norm(beta/(1e-10+np.linalg.norm(beta)) - beta_min_pop_norm_test))

	    
	#Misclassification
	len_X_pop_text    = X_pop_test.shape[0]
	ones_N_test       = np.ones(len_X_pop_text)
	### STANDARDIZATION ###
	misclassification = y_pop_test*np.sign([np.dot(X_pop_test, l2_X_train*beta[0]) + beta[1]*ones_N_test for beta in argmin_betas])
	misclassification = [np.sum(misclassification[i]<0)/float(len_X_pop_text) for i in range(4)]


	#Sparsity and True positive
	supports     = [np.where(beta[0]!=0)[0] for beta in argmin_betas]
	sparsity     = [len(support) for support in supports]
	true_support = np.where(u_positive!=0)[0] 
	true_positive= [len(set(true_support) & set(support)) for support in supports]



#---Plots
	bars_metrics(l2_estimation, 'l2_estimation')
	plt.savefig(pathname+'/'+name_metric+'/test_l2_estimation.pdf')

	bars_metrics(misclassification, 'misclassification')
	plt.savefig(pathname+'/'+name_metric+'/test_misclassification.pdf')

	bars_metrics(sparsity, 'sparsity')
	plt.savefig(pathname+'/'+name_metric+'/test_sparsity.pdf')

	bars_metrics(true_positive, 'true_positive')
	plt.savefig(pathname+'/'+name_metric+'/test_true_positive.pdf')

	return l2_estimation, misclassification, sparsity, true_positive
