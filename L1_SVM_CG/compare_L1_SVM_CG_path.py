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






def compare_L1_SVM_CG_path(type_Sigma, N, P_list, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_CG_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS
	epsilon_RC  = 0
	loop_repeat = 2
	
	n_alpha_list = 25
	time_limit   = 30  



#---RESULTS
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_delete        = [[ [] for i in range(n_alpha_list)] for P in P_list]  #delete the columns not in the support
	times_SVM_CG_no_delete     = [[ [] for i in range(n_alpha_list)] for P in P_list]
	

	size_first_support = [[ [] for i in range(2)] for P in P_list] #get a sense of first support to adjust n_features
	n_features         = 10

	same_supports = [[ [] for i in range(n_alpha_list)] for P in P_list]
	compare_norms = [[ [] for i in range(n_alpha_list)] for P in P_list]
	compare_ratio = [[ [] for i in range(n_alpha_list)] for P in P_list]



	aux_P = -1
	for P in P_list:

		write_and_print('\n\n\n-----------------------------For P='+str(P)+'--------------------------------------', f)

		aux_P += 1
		for i in range(loop_repeat):

			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)


		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
			alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)] #start with non empty support
			#alpha_list   = [1e-4*alpha_max]

			aux_alpha = -1


			index_SVM_CG_delete    = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
			index_SVM_CG_no_delete = init_correlation(X_train, y_train, n_features, f)


			model_L1_SVM           = 0
			model_SVM_CG_delete    = 0
			model_SVM_CG_no_delete = 0


			for alpha in alpha_list:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM with Gurobi without CG #####', f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _ = L1_SVM_CG(X_train, y_train, range(P), alpha, epsilon_RC, time_limit, model_L1_SVM, False, f) #_ = range(P) 
				times_L1_SVM[aux_P][aux_alpha].append(time_L1_SVM)


			#---R penalizedSVM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_penalizedSVM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)

			#---R SAM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: SAM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_SAM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)



			#---L1 SVM with CG and deleting
				write_and_print('\n###### L1 SVM with CG and deletion #####', f)
				beta_SVM_CG, support_SVM_CG, time_SVM_CG, model_SVM_CG_delete, index_SVM_CG_delete   = L1_SVM_CG(X_train, y_train, index_SVM_CG_delete, alpha, epsilon_RC, time_limit, model_SVM_CG_delete, True, f)
				times_SVM_CG_delete[aux_P][aux_alpha].append(time_SVM_CG)
				print len(index_SVM_CG_delete)


			#---L1 SVM with CG and not deleting
				write_and_print('\n###### L1 SVM with CG and no deletion #####', f)
				beta_SVM_CG, support_SVM_CG, time_SVM_CG, model_SVM_CG_no_delete, index_SVM_CG_no_delete = L1_SVM_CG(X_train, y_train, index_SVM_CG_no_delete, alpha, epsilon_RC, time_limit, model_SVM_CG_no_delete, False, f)
				times_SVM_CG_no_delete[aux_P][aux_alpha].append(time_SVM_CG)
				print len(index_SVM_CG_no_delete)


				same_support = len(set(support_SVM_CG)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CG))==0
				same_supports[aux_P][aux_alpha].append(same_support)
				compare_norms[aux_P][aux_alpha].append(np.linalg.norm(beta_SVM_CG - beta_L1_SVM)**2)
				compare_ratio[aux_P][aux_alpha].append(time_L1_SVM/time_SVM_CG)


			#---Get a sense of size of first support to adjust n
				if aux_alpha<2:
					size_first_support[aux_P][aux_alpha].append(len(support_SVM_CG))


			#support = np.array(support)
			#print [set(support[i]) < set(support[i+1]) for i in range(len(support) -1)]



	write_and_print('\nSame support:       '+str(same_supports), f)
	write_and_print('\nSize first support: '+str(size_first_support), f)


#---Ratio of win
	compare_ratio = np.array([ [ np.mean([times_L1_SVM[P][i][loop] for loop in range(loop_repeat) ]) for i in range(n_alpha_list)] for P in range(len(P_list))])
	compare_ratios(type_Sigma, N, P_list, k0, rho, tau_SNR, compare_ratio, alpha_list)
	plt.savefig(pathname+'/compare_ratios.png')
	plt.close()

	#write_and_print('\nNorm difference: '+str(compare_ratio), f)



#---Ratio of win
	times_L1_SVM_averaged              = [ [ np.sum([times_L1_SVM[P][i][loop]              for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_SVM_CG_averaged_delete       = [ [ np.sum([times_SVM_CG_delete[P][i][loop]       for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_SVM_CG_averaged_no_delete    = [ [ np.sum([times_SVM_CG_no_delete[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]

	L1_SVM_CG_plots_path(type_Sigma, N, P_list, k0, rho, tau_SNR, times_L1_SVM_averaged, times_SVM_CG_averaged_delete, times_SVM_CG_averaged_no_delete)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()




