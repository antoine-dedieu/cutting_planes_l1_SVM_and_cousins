import numpy as np
import datetime
import os
import sys

from sklearn.feature_selection import RFE

from gurobipy import *
from L1_SVM_CP import *
from init_L1_SVM_CP import *
from L1_SVM_CP_plots import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../L1_SVM_CG')
from R_L1_SVM import *




def compare_L1_SVM_CP(type_Sigma, N_list, P, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_CP_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS

	epsilon_RC  = 1e-2
	loop_repeat = 5
	
	n_alpha_list = 1
	time_limit   = 30 


#---RESULTS
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_L1_SVM_WS            = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for N in N_list]  #delete the columns not in the support
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for N in N_list]

	
	size_first_support = [[ [] for i in range(2)] for N in N_list] #get a sense of first support to adjust n_features
	n_samples          = P/2

	same_supports = [[ [] for i in range(n_alpha_list)] for N in N_list]
	compare_norms = [[ [] for i in range(n_alpha_list)] for N in N_list]


	aux_N=-1
	for N in N_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+'--------------------------------------', f)

		aux_N += 1
		for i in range(loop_repeat):
			write_and_print('\n\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
			#alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)] #start with non empty support
			alpha_list   = [1e-2*alpha_max]

			aux_alpha = -1


		#---WE DO NOT GIVE A NUMBER OF SAMPLES !!
			index_SVM_CP_L1_norm, time_L1_norm              = init_CP_norm_samples(X_train, y_train, n_samples, f) #just for highest alpha
			idx_liblinear, time_liblinear, beta_liblinear   = liblinear_for_CP('squared_hinge_l1', X_train, y_train, alpha_list[0], f)


			model_L1_SVM_no_WS= 0
			model_L1_SVM_WS   = 0
			model_method_1    = 0
			model_method_2    = 0


			for alpha in alpha_list[::-1]:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)
				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM without CP without warm start #####', f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _, _ = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM_no_WS, [], False, f)
				times_L1_SVM[aux_N][aux_alpha].append(time_L1_SVM)

			#---L1 SVM 
				write_and_print('\n###### L1 SVM without CP with warm start #####', f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _, _ = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM_WS, beta_liblinear, False, f)
				times_L1_SVM_WS[aux_N][aux_alpha].append(time_liblinear + time_L1_SVM)




			#---L1 SVM with cutting planes and deleting
				write_and_print('\n###### L1 SVM with CP L1 norm, eps=1e-2 #####', f)
				beta_method_1, support_method_1, time_method_1, model_method_1, index_SVM_CG_correl, obj_val_method_1 = L1_SVM_CP(X_train, y_train, index_SVM_CP_L1_norm, alpha, 1e-2, time_limit, model_method_1, [], False, f)
				times_SVM_CG_method_1[aux_N][aux_alpha].append(time_L1_norm+time_method_1)


			#---L1 SVM with cutting planes and non deleting
				write_and_print('\n###### L1 SVM with CP liblinear, eps=1e-2 #####', f)
				beta_method_2, support_method_2, time_method_2, model_method_2, index_SVM_CG_liblinear, obj_val_method_2 = L1_SVM_CP(X_train, y_train, idx_liblinear, alpha, 1e-2, time_limit, model_method_2, beta_liblinear, False, f)
				times_SVM_CG_method_2[aux_N][aux_alpha].append(time_liblinear+time_method_2)



				#same_support = len(set(support_SVM_CP)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CP))==0
				#same_supports[aux_N][aux_alpha].append(same_support)

				

	#write_and_print('\nSame support:       '+str(same_supports), f)



#---Compare times
	times_L1_SVM   = [ [ np.sum([times_L1_SVM[N][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_L1_SVM_WS= [ [ np.sum([times_L1_SVM_WS[N][i][loop]          for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_method_1 = [ [ np.sum([times_SVM_CG_method_1[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_method_2 = [ [ np.sum([times_SVM_CG_method_2[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]

	L1_SVM_CP_plots_path(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM, times_method_1, times_method_2)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()

	L1_SVM_CP_plots_errorbar_path(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM, times_L1_SVM_WS, times_method_1, times_method_2, 'time')
	plt.savefig(pathname+'/compare_times_errorbar.png')
	plt.close()



