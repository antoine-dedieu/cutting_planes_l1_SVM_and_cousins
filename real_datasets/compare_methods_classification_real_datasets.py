import numpy as np

import datetime
import os
import sys


from process_data_real_datasets import *
from process_Rahul_real_datasets import *
from process_data_real_datasets_uci import *

from validation_metric_real_datasets import *
from plot_test_metrics_real_datasets import *

sys.path.append('../algorithms')
from heuristics_classification import *


sys.path.append('../synthetic_datasets')
from power_method import *


sys.path.append('../graphics')
from plot_validation_metrics import *
from boxplot_averaged_metrics import *




def compare_methods_classification_real_datasets(type_loss, type_real_dataset, pathname):

#1: LUNG CANCER
#2: LEUKEMIA
#3: RADSENS
#4: ARCENE


#---PARAMETERS

	#Computation parameters
	K0_list   = range(15)
	epsilon   = 1e-3
	N_alpha   = 100 if type_loss=='logreg' else 50#10-4
	number_NS = 2


#---SPLIT DATA
	f = open(pathname+'/results.txt', 'w')

	dict_funtion = {1: split_real_dataset_bis, 2: split_real_dataset_bis, 3:split_Rahul_real_dataset_bis, 4: split_real_dataset_uci_bis}
	X_train, X_test, y_train, y_test, seed, l2_X_train = dict_funtion[type_real_dataset](type_real_dataset, f)


#---SIMULATE DATA
	pathname +=  '/seed_'+str(seed)
	os.makedirs(pathname)
	X_train = np.array([X_train[i][:100] for i in range(X_train.shape[0])])
	X_test  = np.array([X_test[i][:100]  for i in range(X_test.shape[0])])
	l2_X_train = l2_X_train[:100]


#---Parameters
	N, P        = X_train.shape	
	X_add       = 1./math.sqrt(N)*np.ones((N,P+1))
	X_add[:,:P] = X_train
	mu_max      = power_method(X_train)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


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

	argmin_betas, best_k_plot, all_k_plot = validation_metric_real_datasets(betas, train_errors, X_test, y_test, l2_X_train, K0_list, N_alpha)

	compare_best_k(best_k_plot, 'misclassification')
	plt.savefig(pathname+'/best_k.pdf')

	compare_all_k(all_k_plot, K0_list, 'misclassification')
	plt.savefig(pathname+'/all_k.pdf')


	misclassification, sparsity  = plot_test_metrics_real_datasets(argmin_betas, X_test, y_test, l2_X_train, pathname)
	metrics.append(misclassification)
	metrics.append(sparsity)

	plt.close('all')
	f.close()
	return metrics




def average_simulations_compare_methods_classification_real_datasets(type_loss, type_real_dataset, n_average):

	DT = datetime.datetime.now()
	dict_name_dataset = {1:'lung_cancer', 2:'leukemia', 3:'radsens', 4:'arcene'} 
	pathname = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'_'+dict_name_dataset[type_real_dataset]+'_'+type_loss
	pathname = r'../../real_datasets_results/'+type_loss+'/'+str(pathname)
	os.makedirs(pathname)

#---SIMULATIONS
	metrics_to_average = [compare_methods_classification_real_datasets(type_loss, type_real_dataset, pathname) for i in range(n_average)]

#---PLOTS and AVERAGE
	if n_average>1:

		for aux_metric in range(2):
			name_metric     = ['misclassification', 'sparsity'][aux_metric]
			metric_averaged = np.array([ metrics_to_average[i][aux_metric] for i in range(n_average) ]) 

			boxplot_averaged_metrics(metric_averaged,  name_metric)
			plt.savefig(pathname+'/'+name_metric+'.pdf')




