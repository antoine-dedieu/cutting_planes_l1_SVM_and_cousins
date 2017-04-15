import numpy as np
import datetime
import os
import sys

from gurobipy import *

from L1_SVM_both_CG_CP_plots import *
from L1_SVM_add_columns_delete_samples import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *
from R_L1_SVM import *

sys.path.append('../L1_SVM_CP')
from L1_SVM_CP import *
from init_L1_SVM_CP import *




def compare_L1_SVM_both_CG_CP_path(type_Sigma, N_P_list, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_both_CG_CP_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS
	epsilon_RC  = 1e-1
	loop_repeat = 5

	n_alpha_list = 20
	time_limit   = 60 
	n_features   = 10



#---Results
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_SVM_CG_no_delete     = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_SVM_add_delete       = [[ [] for i in range(n_alpha_list)] for P in N_P_list]  #delete the columns not in the support

	objvals_SVM_CG_no_delete    = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	objvals_SVM_add_delete      = [[ [] for i in range(n_alpha_list)] for P in N_P_list] 

	#same_supports    = [[] for N,P in N_P_list]
	#compare_norms    = [[] for N,P in N_P_list]


	aux=-1
	for N,P in N_P_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+' and P='+str(P)+'--------------------------------------', f)

		aux += 1
		for i in range(loop_repeat):
			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
			alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)] #start with non empty support
			delete_samples = True

			aux_alpha = -1


		#---Warm start Gurobi
			index_SVM_CG_no_delete = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
			index_columns          = init_correlation(X_train, y_train, n_features, f)
			index_samples          = range(N) #all constraints for lambda max


			model_L1_SVM           = 0
			model_SVM_CG_no_delete = 0
			model_SVM_add_delete   = 0


			for alpha in alpha_list:
				aux_alpha += 1
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)


			#---L1 SVM 
				write_and_print('\n\n###### L1 SVM#####', f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, objval_L1_SVM = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM, False, f)
				times_L1_SVM[aux][aux_alpha].append(time_L1_SVM/N)


			#---L1 SVM with CG and not deleting
				write_and_print('\n\n###### L1 SVM with CG and no deletion #####', f)
				beta_SVM, support_SVM, time_SVM_CG_no_delete, model_SVM_CG_no_delete, index_SVM_CG_no_delete, objval_SVM_CG_no_delete = L1_SVM_CG(X_train, y_train, index_SVM_CG_no_delete, alpha, epsilon_RC, time_limit, model_SVM_CG_no_delete, False, f)
				times_SVM_CG_no_delete[aux][aux_alpha].append(time_SVM_CG_no_delete/N)
				objvals_SVM_CG_no_delete[aux][aux_alpha].append(objval_SVM_CG_no_delete/objval_L1_SVM)
				print(N, len(index_SVM_CG_no_delete))


			#---L1 SVM with CG and not deleting
				write_and_print('\n\n###### L1 SVM with column generation and constraint deletion #####', f)
				beta_SVM, support_SVM, time_SVM_add_delete, model_SVM_add_delete, index_samples, index_columns, delete_samples, objval_add_delete = L1_SVM_add_columns_delete_samples(X_train, y_train, index_samples, index_columns, alpha, epsilon_RC, time_limit, model_SVM_add_delete, delete_samples, f)
				times_SVM_add_delete[aux][aux_alpha].append(time_SVM_add_delete/N)
				objvals_SVM_add_delete[aux][aux_alpha].append(objval_add_delete/objval_L1_SVM)
				print len(index_samples), len(index_columns) 



			#---Store
				#same_support = len(set(support_SVM_CG)-set(support_R_L1_SVM))==0 and len(set(support_R_L1_SVM)-set(support_SVM_CG))==0
				#same_supports[aux].append(same_support)


	#write_and_print('\nSame support: '+str(same_supports), f)

#---TIMES
	times_L1_SVM_averaged              = [ [ np.sum([times_L1_SVM[NP][i][loop]              for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for NP in range(len(N_P_list))]
	times_SVM_CG_averaged_no_delete    = [ [ np.sum([times_SVM_CG_no_delete[NP][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for NP in range(len(N_P_list))]
	times_SVM_CG_averaged_add_delete   = [ [ np.sum([times_SVM_add_delete[NP][i][loop]      for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for NP in range(len(N_P_list))]


	times_L1_SVM_both_CG_CP_plots_path(type_Sigma, N_P_list, k0, rho, tau_SNR, times_L1_SVM_averaged, times_SVM_CG_averaged_no_delete, times_SVM_CG_averaged_add_delete)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()


#---RATIO
	objvals_SVM_CG_no_delete_averaged    = [ [ np.mean([objvals_SVM_CG_no_delete[NP][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for NP in range(len(N_P_list))]
	objvals_SVM_CG_add_delete_averaged   = [ [ np.mean([objvals_SVM_add_delete[NP][i][loop]      for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for NP in range(len(N_P_list))]

	objval_L1_SVM_both_CG_CP_plots_path(type_Sigma, N_P_list, k0, rho, tau_SNR, objvals_SVM_CG_no_delete_averaged, objvals_SVM_CG_add_delete_averaged)
	plt.savefig(pathname+'/compare_objective_values.png')
	plt.close()




