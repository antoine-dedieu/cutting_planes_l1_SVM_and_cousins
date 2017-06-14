import numpy as np
import pandas as pd

import datetime
import os
import sys

sys.path.append('../algorithms')
from heuristics_classification import *
from neighborhood_search_classification import *




from simulate_data_classification import *

from heuristic_VS_rfe import *
from statistical_graphics import *

from neighborhood_search_tools import *

from MIO_tools_classification import *







def compare_methods_classification(type_loss, N, P, k0, rho, tau, type_Sigma):


#---PARAMETERS
	#Random parameters
	seed_X = random.randint(0,1000)


	#Computation parameters
	K0_list   = range(15)
	epsilon   = 1e-3
	N_alpha   = 50 #10-4
	number_NS = 2




#---SIMULATE DATA
	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau)+'_Sigma'+str(type_Sigma)+'_seed_X_'+str(seed_X)
	pathname=r'../../synthetic_datasets_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')

	X_train, X_test, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau, seed_X, f)


	#POP HINGE LOSS
	X_train, X_test, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, 10000, P, k0, rho, tau, seed_X, f)


	#Algo1 parameters
	X_add       = np.ones((N,P+1))
	X_add[:,:P] = X_train
	mu_max      = power_method(X_train)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#---RFE
	train_errors_K0_list_RFE_l2, betas_K0_list_RFE_l2 = all_RFE_estimators(type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_RFE_l1, betas_K0_list_RFE_l1 = all_RFE_estimators(type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)


	
#---HEURISTIC 3

	#l2
	train_errors_K0_list_H3_l2,  betas_K0_list_H3_l2, _, _   = heuristic3('up',   type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_H3b_l2, betas_K0_list_H3b_l2, _, _  = heuristic3('down', type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	
	best_train_errors_H3_l2, best_betas_H3_l2 = best_of_up_down(train_errors_K0_list_H3_l2, train_errors_K0_list_H3b_l2, betas_K0_list_H3_l2, betas_K0_list_H3b_l2, K0_list, number_decreases, f)


	#l1
	train_errors_K0_list_H3_l1,  betas_K0_list_H3_l1, _, _   = heuristic3('up',   type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_H3b_l1, betas_K0_list_H3b_l1, _, _  = heuristic3('down', type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	
	best_train_errors_H3_l1, best_betas_H3_l1 = best_of_up_down(train_errors_K0_list_H3_l1, train_errors_K0_list_H3b_l1, betas_K0_list_H3_l1, betas_K0_list_H3b_l1, K0_list, number_decreases, f)    




#---COMPARE

	#Best test errors
	heuristic_VS_rfe(    type_loss, X_train, X_test, y_train, y_test, u_positive, best_betas_H3_l2, best_betas_H3_l1, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, f)
	
	#Plots
	statistical_graphics(type_loss, X_train, X_test, y_train, y_test, u_positive, best_betas_H3_l2, best_betas_H3_l1, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, N,P,rho,tau)
	plt.savefig(pathname+'/compare_penalizations_after_heuristic.png')



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#---NEIGHBORHOOD SEARCH 
	
	#l2: randomized and determinsitic
	train_errors_l2_randomized,    betas_l2_randomized    = np.copy(best_train_errors_H3_l2), np.copy(best_betas_H3_l2)
	train_errors_l2_deterministic, betas_l2_deterministic = np.copy(best_train_errors_H3_l2), np.copy(best_betas_H3_l2)


	#l1: randomized and determinsitic
	train_errors_l1_randomized,    betas_l1_randomized    = np.copy(best_train_errors_H3_l1), np.copy(best_betas_H3_l1)
	train_errors_l1_deterministic, betas_l1_deterministic = np.copy(best_train_errors_H3_l1), np.copy(best_betas_H3_l1)



	for number_NS in range(number_neighborhood_search):
		#l2
		train_errors_l2_randomized,    betas_l2_randomized, _    = randomized_NS(   number_NS, type_loss, 'l2', X_train, y_train, best_train_errors_H3_l2, [], train_errors_l2_randomized,    betas_l2_randomized,    K0_list, number_decreases, alpha_list_l2, X_add, L, epsilon, time_limit, f)    
		train_errors_l2_deterministic, betas_l2_deterministic, _ = deterministic_NS(number_NS, type_loss, 'l2', X_train, y_train, best_train_errors_H3_l2, [], train_errors_l2_deterministic, betas_l2_deterministic, K0_list, number_decreases, alpha_list_l2, X_add, L, epsilon, time_limit, f)    


		#l2
		train_errors_l1_randomized,    betas_l1_randomized, _    = randomized_NS(   number_NS, type_loss, 'l1', X_train, y_train, best_train_errors_H3_l1, [], train_errors_l1_randomized,    betas_l1_randomized,    K0_list, number_decreases, alpha_list_l1, X_add, L, epsilon, time_limit, f)    
		train_errors_l1_deterministic, betas_l1_deterministic, _ = deterministic_NS(number_NS, type_loss, 'l1', X_train, y_train, best_train_errors_H3_l1, [], train_errors_l1_deterministic, betas_l1_deterministic, K0_list, number_decreases, alpha_list_l1, X_add, L, epsilon, time_limit, f)    





#---COMPARE
	#Average train erros
	averaged_objective_value_after_NS('l2', train_errors_K0_list_RFE_l2, train_errors_K0_list_H1_l2, best_train_errors_H2_l2, best_train_errors_H3_l2, train_errors_l2_randomized, train_errors_l2_deterministic, K0_list, number_decreases, f)
	averaged_objective_value_after_NS('l1', train_errors_K0_list_RFE_l1, train_errors_K0_list_H1_l1, best_train_errors_H2_l1, best_train_errors_H3_l1, train_errors_l1_randomized, train_errors_l1_deterministic, K0_list, number_decreases, f)


	#Best test errors
	heuristic_VS_rfe(    type_loss, X_train, X_test, y_train, y_test, u_positive, betas_l2_randomized, betas_l1_randomized, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, f)
	
	#Plots
	statistical_graphics(type_loss, X_train, X_test, y_train, y_test, u_positive, betas_l2_randomized, betas_l1_randomized, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, N, P, rho, tau)
	plt.savefig(pathname+'/compare_penalizations_after_NS.png')



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#---GUROBI 
	
	#We only run MIO for a few K0
	len_real_support = len(np.where(u_positive!=0)[0])
	K0_list_MIO      = range(int(len_real_support)-2, int(len_real_support)+4)


	train_errors_l2_Gurobi, betas_l2_Gurobi, _, index_support_l2, VS_before_after_l2, _ = MIO_improvements(type_loss, 'l2', X_train, y_train, u_positive, train_errors_l2_randomized, betas_l2_randomized, alpha_list_l2, K0_list_MIO, number_decreases, time_limit, f)
	train_errors_l1_Gurobi, betas_l1_Gurobi, _, index_support_l1, VS_before_after_l1, _ = MIO_improvements(type_loss, 'l1', X_train, y_train, u_positive, train_errors_l1_randomized, betas_l1_randomized, alpha_list_l1, K0_list_MIO, number_decreases, time_limit, f)
	


#---COMPARE RESULTS
	#averaged_objective_value_after_MIO('l2', train_errors_K0_list_RFE_l2, train_errors_K0_list_H1_l2, best_train_errors_H3_l2, train_errors_l2_randomized, train_errors_l2_Gurobi, K0_list_MIO, index_support_l2, f)
	#averaged_objective_value_after_MIO('l1', train_errors_K0_list_RFE_l1, train_errors_K0_list_H1_l1, best_train_errors_H3_l1, train_errors_l1_randomized, train_errors_l1_Gurobi, K0_list_MIO, index_support_l1, f)
	

	#Best test errors
	#heuristic_VS_rfe(    type_loss, X_train, X_test, y_train, y_test, u_positive, betas_l2_Gurobi, betas_l1_Gurobi, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list_MIO, f)
	
	#Plots
	#statistical_graphics(type_loss, X_train, X_test, y_train, y_test, u_positive, betas_l2_Gurobi, betas_l1_Gurobi, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list_MIO, N,P,rho,tau)
	#plt.savefig(pathname+'/compare_penalizations_after_MIO.png')

	


#---CLOSE THE FILE
	f.close()





