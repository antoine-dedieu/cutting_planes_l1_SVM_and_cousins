import numpy as np
import sys

sys.path.append('../graphics')
from bars_metrics import *


def plot_test_metrics_real_datasets(argmin_betas, X_test, y_test, l2_X_train, pathname):
	
#---Metrics
	#Misclassification
	len_X_text    = X_test.shape[0]
	ones_N_test   = np.ones(len_X_text)
	### STANDARDIZATION ###
	misclassification = y_test*np.sign([np.dot(X_test, l2_X_train*beta[0]) + beta[1]*ones_N_test for beta in argmin_betas])
	misclassification = [np.sum(misclassification[i]<0)/float(len_X_text) for i in range(4)]


	#Sparsity and True positive
	supports     = [np.where(beta[0]!=0)[0] for beta in argmin_betas]
	sparsity     = [len(support) for support in supports]


#---Plots
	bars_metrics(misclassification, 'misclassification')
	plt.savefig(pathname+'/test_misclassification.pdf')

	bars_metrics(sparsity, 'sparsity')
	plt.savefig(pathname+'/test_sparsity.pdf')

	return misclassification, sparsity[:3]
