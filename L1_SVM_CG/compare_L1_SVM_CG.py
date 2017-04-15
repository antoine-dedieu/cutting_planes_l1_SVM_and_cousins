import numpy as np
import datetime
import time
import os
import sys


from gurobipy import *
from L1_SVM_CG import *
from L1_SVM_CG_plots import *

from R_L1_SVM import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *
from R_L1_SVM import *






def compare_L1_SVM_CG(type_Sigma, N, P_list, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_CG_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS
	epsilon_RC  = 1e-2
	loop_repeat = 5
	n_features  = N

	alpha_max    = np.max(np.sum( np.abs(X_train), axis=0)) #infinite norm of the sum over the lines
	n_alpha_list = 50
	alpha_list   = [alpha_max*0.8**i for i in np.arange(n_alpha_list)]

	#alpha_list   = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10][::-1] #decreasing order -> high enough for not being affected by epsilon_RC
	#alpha_list   = [1e-1]
	time_limit   = 30  



#---RESULTS
	times_L1_SVM               = [[ [] for i in alpha_list] for P in P_list]
	times_penalizedSVM_R_SVM   = [[ [] for i in alpha_list] for P in P_list]
	times_SAM_R_SVM            = [[ [] for i in alpha_list] for P in P_list]
	times_SVM_CG               = [[ [] for i in alpha_list] for P in P_list]
	
	same_supports = [[ [] for i in alpha_list] for P in P_list]
	compare_norms = [[ [] for i in alpha_list] for P in P_list]
	compare_ratio = [[ [] for i in alpha_list] for P in P_list]



	aux_P = -1
	for P in P_list:

		write_and_print('\n\n\n-----------------------------For P='+str(P)+'--------------------------------------', f)

		aux_P += 1
		for i in range(loop_repeat):

			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)


			aux_alpha = -1

			index_SVM_CG = init_correlation(X_train, y_train, n_features, f) #independent of alpha

			model_L1_SVM = 0
			model_SVM_CG = 0
			#support = []


			for alpha in alpha_list:

				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM with Gurobi without CG #####', f)
				#beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, index_L1_SVM = L1_SVM_CG(X_train, y_train, index_SVM_CG, alpha, epsilon_RC, time_limit, model_L1_SVM, f) #index_L1_SVM still = range(N) 
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, _ = L1_SVM_CG(X_train, y_train, range(P), alpha, epsilon_RC, time_limit, model_L1_SVM, f) #_ = range(P) 
				
				times_L1_SVM[aux_P][aux_alpha].append(time_L1_SVM)
				#support.append(support_L1_SVM)


			#---R penalizedSVM L1 SVM 
				write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
				beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
				times_penalizedSVM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)



			#---R SAM L1 SVM 
				write_and_print('\n###### L1 SVM with R: SAM #####', f)
				beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
				times_SAM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)



			#---L1 SVM with CG
				write_and_print('\n###### L1 SVM with CG #####', f)
				#index_CG = RFE_for_CG(X_train, y_train, alpha, n_features, f)
				
				beta_SVM_CG, support_SVM_CG, time_SVM_CG, model_SVM_CG, index_SVM_CG, _ = L1_SVM_CG(X_train, y_train, index_SVM_CG, alpha, epsilon_RC, time_limit, model_SVM_CG, f)
				times_SVM_CG[aux_P][aux_alpha].append(time_SVM_CG)


				same_support = len(set(support_SVM_CG)-set(support_R_L1_SVM))==0 and len(set(support_R_L1_SVM)-set(support_SVM_CG))==0
				same_supports[aux_P][aux_alpha].append(same_support)
				compare_norms[aux_P][aux_alpha].append(np.linalg.norm(beta_SVM_CG - beta_R_L1_SVM)**2)

				compare_ratio[aux_P][aux_alpha].append(time_L1_SVM/time_SVM_CG)


			#support = np.array(support)
			#print [set(support[i]) < set(support[i+1]) for i in range(len(support) -1)]



	write_and_print('\nSame support: '+str(same_supports), f)

	compare_ratio = np.array([ [ np.mean([times_L1_SVM[P][i][loop] for loop in range(loop_repeat) ]) for i in range(n_alpha_list)] for P in range(len(P_list))])
	compare_ratios(type_Sigma, N, P_list, k0, rho, tau_SNR, compare_ratio, alpha_list)
	plt.savefig(pathname+'/compare_ratios.png')
	plt.close()

	#write_and_print('\nNorm difference: '+str(compare_ratio), f)


	times_L1_SVM_averaged              = [ [ np.mean([times_L1_SVM[P][i][loop]              for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_penalizedSVM_R_SVM_averaged  = [ [ np.mean([times_penalizedSVM_R_SVM[P][i][loop]  for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_SAM_R_SVM_averaged           = [ [ np.mean([times_SAM_R_SVM[P][i][loop]           for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_SVM_CG_averaged              = [ [ np.mean([times_SVM_CG[P][i][loop]              for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]

	L1_SVM_CG_plots(type_Sigma, N, P_list, k0, rho, tau_SNR, times_L1_SVM_averaged, times_penalizedSVM_R_SVM_averaged, times_SAM_R_SVM_averaged, times_SVM_CG_averaged)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()

	#return times_L1_SVM, times_SVM_CG, same_supports



