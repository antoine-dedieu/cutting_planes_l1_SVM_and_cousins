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
from L1_SVM_CG_plots import *




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
	loop_repeat = 3
	
	n_alpha_list = 1
	time_limit   = 30 


#---RESULTS
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for N in N_list] 
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CG_method_3      = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CG_method_4      = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CG_method_5      = [[ [] for i in range(n_alpha_list)] for N in N_list]



	aux_N=-1
	for N in N_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+'--------------------------------------', f)

		aux_N += 1
		for i in range(loop_repeat):
			write_and_print('\n\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))           
			alpha_list   = [1e-2*alpha_max]
			aux_alpha = -1


		#---TRY WITH AND WTHOUT RESTRICTION
			index_samples_int, time_init                       = init_CP_sampling_smoothing(X_train, y_train, alpha_list[0], False, f)
			index_samples_int_restricted, time_init_restricted = init_CP_sampling_smoothing(X_train, y_train, alpha_list[0], True, f)

			idx_liblinear, time_liblinear, beta_liblinear  = liblinear_for_CP('squared_hinge_l1', X_train, y_train, alpha_list[0], f)


			model_L1_SVM_no_WS= 0
			model_L1_SVM_WS   = 0
			model_method_1    = 0
			model_method_2    = 0
			model_method_4    = 0


			for alpha in alpha_list[::-1]:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)
				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM without CP without warm start #####', f)
				if N < 1e2:
					beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _, _ = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM_no_WS, [], f)
				else:
					time_L1_SVM = 1
				times_L1_SVM[aux_N][aux_alpha].append(time_L1_SVM)


			#---L1 SVM with cutting planes and no samples restriction
				write_and_print('\n###### L1 SVM with random subsampling wo restriction, eps=1e-2 #####', f)
				beta_method_1, support_method_1, time_method_1, model_method_1, index_SVM_CG_correl, obj_val_method_1 = L1_SVM_CP(X_train, y_train, index_samples_int, alpha, 1e-2, time_limit, model_method_1, [], f)
				times_SVM_CG_method_1[aux_N][aux_alpha].append(time_init+time_method_1)
				#times_SVM_CG_method_2[aux_N][aux_alpha].append(time_method_1)

			#---L1 SVM with cutting planes and deleting
				write_and_print('\n###### L1 SVM with CP L1 random subsampling with restriction, eps=1e-2 #####', f)
				beta_method_4, support_method_4, time_method_4, model_method_4, index_SVM_CG_correl, obj_val_method_4 = L1_SVM_CP(X_train, y_train, index_samples_int_restricted, alpha, 1e-2, time_limit, model_method_4, [], f)
				times_SVM_CG_method_3[aux_N][aux_alpha].append(time_init_restricted+time_method_4)
				#times_SVM_CG_method_4[aux_N][aux_alpha].append(time_method_4)


			#---L1 SVM with cutting planes and non deleting
				write_and_print('\n###### L1 SVM with CP liblinear, eps=1e-2 #####', f)
				beta_method_3, support_method_3, time_method_3, model_method_3, index_SVM_CG_liblinear, obj_val_method_3 = L1_SVM_CP(X_train, y_train, idx_liblinear, alpha, 1e-2, time_limit, model_method_2, beta_liblinear, f)
				times_SVM_CG_method_5[aux_N][aux_alpha].append(time_liblinear+time_method_3)


				



#---Compare times
	times_L1_SVM   = [ [ np.sum([times_L1_SVM[N][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_method_1 = [ [ np.sum([times_SVM_CG_method_1[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	#times_method_2 = [ [ np.sum([times_SVM_CG_method_2[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_method_3 = [ [ np.sum([times_SVM_CG_method_3[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	#times_method_4 = [ [ np.sum([times_SVM_CG_method_4[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_method_5 = [ [ np.sum([times_SVM_CG_method_5[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]

	#legend_plot_1 = {0:'Gurobi', 1:'FO + CG', 2:'Constraint FO given, CG', 3:'FO restricted + CG', 4:'FO restricted given, CG', 5:'Scikit + CG'}
	legend_plot_1 = {0:'Gurobi', 1:'FO + CG', 2:'FO restricted + CG', 3:'Scikit + CG'}
	
	times_list    = [times_L1_SVM, times_method_1, times_method_3, times_method_5]
	N_list        = [str(int(0.001*N))+'K' for N in N_list]

	L1_SVM_plots_errorbar(type_Sigma, N_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar.pdf', bbox_inches='tight')
	plt.close()



