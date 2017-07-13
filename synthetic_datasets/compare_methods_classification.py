import numpy as np
import datetime
import os
import sys
import math
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from simulate_data_classification_sparsity import *
from power_method import *
from validation_metric import *
from test_metrics import *


sys.path.append('../algorithms')
from heuristics_classification import *

sys.path.append('../graphics')
from plot_validation_metrics import *
from boxplot_averaged_metrics import *
from plots_errorbar_SNR import *





def compare_methods_classification(type_loss, N, P, k0, rho, d_mu, type_Sigma, pathname):


#---PARAMETERS
	#Random parameters
	seed_X = random.randint(0,1000)


	#Computation parameters
	K0_list   = range(15)
	epsilon   = 1e-3
	N_alpha   = 100 if type_loss=='logreg' else 50#10-4
	number_NS = 0



#---SIMULATE DATA
	pathname += '/seedX_'+str(seed_X)
	os.makedirs(pathname)
	f = open(pathname+'/results.txt', 'w')

	X_train, _, l2_X_train, y_train, _, u_positive = simulate_train_test_classification_sparsity(type_Sigma, N, P, k0, rho, d_mu, seed_X, f)


#---POP HINGE LOSS
	X_pop_val, X_pop_test, l2_X_pop_val, y_pop_val, y_pop_test, u_positive = simulate_train_test_classification_sparsity(type_Sigma, 10000, P, k0, rho, d_mu, seed_X, f)

	#Standardization
	l2_X_pop_test = [np.linalg.norm(X_pop_test[:,i]) for i in range(P)]
	for i in range(P): X_pop_test[:,i] /= l2_X_pop_test[i]

	#Estimators
	beta_min_pop_val,  _, _ = estimator_on_support(type_loss, 'l2', X_pop_val,  y_pop_val,  1e-7, u_positive)
	beta_min_pop_test, _, _ = estimator_on_support(type_loss, 'l2', X_pop_test, y_pop_test, 1e-7, u_positive)

	#Standardization
	beta_min_pop_val  = beta_min_pop_val*l2_X_pop_val
	beta_min_pop_test = beta_min_pop_test*l2_X_pop_test

	#We dont want val and test normalized
	for i in range(P):
	    X_pop_val[:,i]  *= l2_X_pop_val[i]
	    X_pop_test[:,i] *= l2_X_pop_test[i]



#---Algo1 parameters
	X_add       = 1./math.sqrt(N)*np.ones((N,P+1))
	X_add[:,:P] = X_train
	mu_max      = power_method(X_train)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


	
#---HEURISTIC 3
	#l1
	betas_up,   train_errors_up,   alphas_l1, _, _                      = heuristic('up',   type_loss, 'l1', X_train, y_train, K0_list, N_alpha, X_add, mu_max, epsilon, f)
	betas_down, train_errors_down, _, betas_l1_SVM, train_errors_l1_SVM = heuristic('down', type_loss, 'l1', X_train, y_train, K0_list, N_alpha, X_add, mu_max, epsilon, f)

	betas_l1, train_errors_l1 = best_of_up_down(train_errors_up, train_errors_down, betas_up, betas_down, K0_list, N_alpha, f) 


	#l2
	betas_up,   train_errors_up,  alphas_l2, _, _                       = heuristic('up',   type_loss, 'l2', X_train, y_train, K0_list, N_alpha, X_add, mu_max, epsilon, f)
	betas_down, train_errors_down, _, betas_l2_SVM, train_errors_l2_SVM = heuristic('down', type_loss, 'l2', X_train, y_train, K0_list, N_alpha, X_add, mu_max, epsilon, f)

	betas_l2, train_errors_l2 = best_of_up_down(train_errors_up, train_errors_down, betas_up, betas_down, K0_list, N_alpha, f)


	
#---COMPARE 
	betas        = [betas_l1, betas_l2, betas_l1_SVM, betas_l2_SVM]
	train_errors = [train_errors_l1, train_errors_l2, train_errors_l1_SVM, train_errors_l2_SVM]
	metrics      = []

	for name_metric_validation in ['l2_estimation', 'misclassification']:
		argmin_betas, best_k_plot, all_k_plot = validation_metric(betas, train_errors, X_pop_val, y_pop_val, beta_min_pop_val, l2_X_train, name_metric_validation, K0_list, N_alpha)
		
		os.makedirs(pathname+'/'+name_metric_validation)

		compare_best_k(best_k_plot, name_metric_validation)
		plt.savefig(pathname+'/'+name_metric_validation+'/best_k.pdf')

		compare_all_k(all_k_plot, K0_list, name_metric_validation)
		plt.savefig(pathname+'/'+name_metric_validation+'/all_k.pdf')
		

		l2_estimation, misclassification, sparsity, true_positive = plot_test_metrics(argmin_betas, X_pop_test, y_pop_test, beta_min_pop_test, l2_X_train, u_positive, pathname, name_metric_validation)
		metrics.append([l2_estimation, misclassification, sparsity, true_positive])
		np.save(pathname+'/'+name_metric_validation+'/metrics', [l2_estimation, misclassification, sparsity, true_positive])

	plt.close('all')
	f.close()
	return metrics





def average_simulations_compare_methods_classification(type_loss, N, P, k0, rho, d_mu, type_Sigma, n_average):

	DT = datetime.datetime.now()
	pathname = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_dmu'+str(d_mu)+'_Sigma'+str(type_Sigma)+'_'+type_loss
	pathname = r'../../synthetic_datasets_results/'+type_loss+'/'+str(pathname)

#---SIMULATIONS
	metrics_to_average = [compare_methods_classification(type_loss, N, P, k0, rho, d_mu, type_Sigma, pathname) for i in range(n_average)]

#---PLOTS and AVERAGE
	if n_average>1:
		for aux_val in range(2):
			name_metric_validation = ['l2_estimation_averaged', 'misclassification_averaged'][aux_val]
			os.makedirs(pathname+'/'+name_metric_validation)

			for aux_metric in range(4):
				name_metric     = ['l2_estimation', 'misclassification', 'sparsity', 'true_positive'][aux_metric]
				metric_averaged = np.array([ metrics_to_average[i][aux_val][aux_metric] for i in range(n_average) ]) 

				boxplot_averaged_metrics(metric_averaged,  name_metric)
				plt.savefig(pathname+'/'+name_metric_validation+'/'+name_metric+'.pdf')




def average_simulations_compare_methods_classification_with_SNR(type_loss, N, P, k0, rho, type_Sigma, n_average):


	DT = datetime.datetime.now()
	pathname = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_Sigma'+str(type_Sigma)+'_'+type_loss
	pathname = r'../../synthetic_datasets_results/'+type_loss+'/'+str(pathname)

	all_metric_averaged = []
	SNR_list            = [0.5, 1, 2, 5, 10, 20, 50]

	N_metrics  = 4
	N_oponents = {0:4, 1:4, 2:3, 3:3}

	for SNR in SNR_list:
		pathname_bis = pathname+'/dmu'+str(SNR)
		metrics_to_average_SNR = [compare_methods_classification(type_loss, N, P, k0, rho, SNR, type_Sigma, pathname_bis) for i in range(n_average)]
		
		aux_val = 1
		metric_averaged_SNR    = np.array([[ metrics_to_average_SNR[simu][aux_val][aux_metric] for aux_metric in range(N_metrics) ] for simu in range(n_average)]) #no more aux_val
		all_metric_averaged.append(metric_averaged_SNR)


	legends        = ['L1+L0', 'L2+L0', 'L1', 'L2']
	name_metrics   = ['l2_estimation', 'misclassification', 'sparsity', 'true_positive']

	for i in range(N_metrics):
		metric_averaged = [[[all_metric_averaged[a][b][i][c] for b in range(n_average)] for a in range(len(SNR_list))] for c in range(N_oponents[i])] 

		#print '\n'
		#print name_metrics[i]
		#print [[all_metric_averaged[a][0][c][3] for c in range(n_average)] for a in range(len(SNR_list))] 
		#print [[all_metric_averaged[a][1][c][3] for c in range(n_average)] for a in range(len(SNR_list))] 
		#print [[all_metric_averaged[a][2][c][3] for c in range(n_average)] for a in range(len(SNR_list))] 

		plots_errorbar_SNR(SNR_list, metric_averaged, legends, name_metrics[i])
		plt.savefig(pathname+'/'+name_metrics[i]+'.pdf', bbox_inches='tight')
		plt.close()

		np.save(pathname+'/metrics_averaged', metric_averaged)








