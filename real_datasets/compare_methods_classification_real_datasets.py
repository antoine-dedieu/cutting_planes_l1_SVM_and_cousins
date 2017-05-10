import numpy as np
import pandas as pd

import datetime
import os
import sys

sys.path.append('../algorithms')
from heuristics_classification import *
from neighborhood_search_classification import *



from process_data_real_datasets import *

from heuristic_VS_rfe_real_datasets import *
from statistical_graphics_real_datasets import *

from neighborhood_search_tools import *
from MIO_tools_classification import *







def compare_methods_classification_real_datasets(type_loss, N, P, k0, rho, tau, type_Sigma):


#---PARAMETERS

	##Computation parameters
	K0_list=range(15)
	epsilon=5e-3
	number_decreases = 50 #10-4
	number_neighborhood_search = 2


	#Gurobi
	time_limit = 20



	#---SIMULATE DATA
	DT = datetime.datetime.now()
	dict_title ={1:'lung_cancer',2:'leukemia'}
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../real_datasets_results/'+str(name)
	os.makedirs(pathname)


	#Open file
	f = open(pathname+'/results.txt', 'w')
	X_train, y_train, X_test, y_test = process_data_real_datasets(type_real_dataset)
	N_train,P = X_train.shape


	write_and_print('DATA CREATED', f)
	write_and_print('Train size : '+str(X_train.shape) + '    +1 train size : '+str((len(y_train) - np.sum(y_train))/2), f)
	write_and_print('Test size  : '+str(X_test.shape)  + '    +1 test size  : '+str((len(y_test) - np.sum(y_test))/2), f)



	#Algo1 parameters
	X_add = np.ones((N_train,P+1))
	X_add[:,:P] = X_train

	L = np.linalg.norm(np.dot(X_add.T,X_add))


	

#---STORE VALUES
	pd.DataFrame(X_train).to_csv(pathname+'/X_train.csv')
	pd.DataFrame(X_test).to_csv(pathname+'/X_test.csv')
	pd.Series(y_train).to_csv(pathname+'/y_train.csv')
	pd.Series(y_test).to_csv(pathname+'/y_test.csv')
	pd.Series(u_positive).to_csv(pathname+'/beta.csv')


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#---RFE
	train_errors_K0_list_RFE_l2, betas_K0_list_RFE_l2 = all_RFE_estimators(type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_RFE_l1, betas_K0_list_RFE_l1 = all_RFE_estimators(type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)



#---HEURISTIC 1
	train_errors_K0_list_H1_l2, betas_K0_list_H1_l2, alpha_list_l2  = heuristic1(type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_H1_l1, betas_K0_list_H1_l2, alpha_list_l1  = heuristic1(type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)



#---HEURISTIC 2

	#l2
	train_errors_K0_list_H2_l2,  betas_K0_list_H2_l2, _, _   = heuristic2('up',   type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_H2b_l2, betas_K0_list_H2b_l2, _, _  = heuristic2('down', type_loss, 'l2', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	
	best_train_errors_H2_l2, best_betas_H2_l2 = best_of_up_down(train_errors_K0_list_H2_l2, train_errors_K0_list_H2b_l2, betas_K0_list_H2_l2, betas_K0_list_H2b_l2, K0_list, number_decreases, f)


	#l1
	train_errors_K0_list_H2_l1,  betas_K0_list_H2_l1, _, _   = heuristic2('up',   type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)
	train_errors_K0_list_H2b_l1, betas_K0_list_H2b_l1, _, _  = heuristic2('down', type_loss, 'l1', X_train, y_train, K0_list, number_decreases, X_add, L, epsilon, time_limit, f)

	best_train_errors_H2_l1, best_betas_H2_l1 = best_of_up_down(train_errors_K0_list_H2_l1, train_errors_K0_list_H2b_l1, betas_K0_list_H2_l1, betas_K0_list_H2b_l1, K0_list, number_decreases, f)



	
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
	heuristic_VS_rfe_real_datasets(    type_loss, X_train, X_test, y_train, y_test, best_betas_H3_l2, best_betas_H3_l1, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, f)
	
	#Plots
	statistical_graphics_real_datasets(type_loss, X_train, X_test, y_train, y_test, best_betas_H3_l2, best_betas_H3_l1, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, N,P,rho,tau)
	plt.savefig(pathname+'/compare_penalizations_after_heuristic.png')



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#---NEIGHBORHOOD SEARCH 
	
	#l2: randomized and determinsitic
	train_errors_l2_randomized,    betas_l2_randomized    = best_train_errors_H3_l2, best_betas_H3_l2
	train_errors_l2_deterministic, betas_l2_deterministic = best_train_errors_H3_l2, best_betas_H3_l2


	#l1: randomized and determinsitic
	train_errors_l1_randomized,    betas_l1_randomized    = best_train_errors_H3_l1, best_betas_H3_l1
	train_errors_l1_deterministic, betas_l1_deterministic = best_train_errors_H3_l1, best_betas_H3_l1



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
	heuristic_VS_rfe_real_datasets(    type_loss, X_train, X_test, y_train, y_test, betas_l2_randomized, betas_l1_randomized, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, f)
	
	#Plots
	statistical_graphics_real_datasets(type_loss, X_train, X_test, y_train, y_test, betas_l2_randomized, betas_l1_randomized, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, N, P, rho, tau)
	plt.savefig(pathname+'/compare_penalizations_after_NS.png')



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#---GUROBI 
	
	#We only run MIO for a few K0
	len_real_support = len(np.where(u_positive!=0)[0])
	K0_list_MIO      = range(int(len_real_support)-2, int(len_real_support)+2)


	train_errors_l2_Gurobi, betas_l2_Gurobi, _, index_support_l2, VS_before_after_l2, _ = MIO_improvements_real_datasets(type_loss, 'l2', X_train, y_train, train_errors_l2_randomized, betas_l2_randomized, K0_list_MIO, number_decreases, alpha_list_l2, time_limit, f)
	train_errors_l1_Gurobi, betas_l1_Gurobi, _, index_support_l1, VS_before_after_l1, _ = MIO_improvements_real_datasets(type_loss, 'l1', X_train, y_train, train_errors_l1_randomized, betas_l1_randomized, K0_list_MIO, number_decreases, alpha_list_l1, time_limit, f)
	


#---COMPARE RESULTS
	averaged_objective_value_after_MIO('l2', train_errors_K0_list_RFE_l2, train_errors_K0_list_H1_l2, best_train_errors_H3_l2, train_errors_l2_randomized, train_errors_L2_heuristic3_Gurobi, K0_list_MIO, index_support_l2, f)
	averaged_objective_value_after_MIO('l1', train_errors_K0_list_RFE_l2, train_errors_K0_list_H1_l2, best_train_errors_H3_l2, train_errors_l2_randomized, train_errors_L2_heuristic3_Gurobi, K0_list_MIO, index_support_l2, f)
	

	
	#Best test errors
	heuristic_VS_rfe_real_datasets(    type_loss, X_train, X_test, y_train, y_test, betas_l2_Gurobi, betas_l1_Gurobi, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, f)
	
	#Plots
	statistical_graphics_real_datasets(type_loss, X_train, X_test, y_train, y_test, betas_l2_Gurobi, betas_l1_Gurobi, betas_K0_list_RFE_l2, betas_K0_list_RFE_l1, K0_list, number_decreases, N,P,rho,tau)
	plt.savefig(pathname+'/compare_penalizations_after_MIO.png')

	


#---CLOSE THE FILE
	f.close()




