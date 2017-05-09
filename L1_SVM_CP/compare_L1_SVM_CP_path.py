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




def compare_L1_SVM_CP_path(type_Sigma, N_list, P, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-P'+str(P)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_CP_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS

	epsilon_RC  = 0
	loop_repeat = 2
	
	n_alpha_list = 25
	time_limit   = 30 


#---RESULTS
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for N in N_list]
	times_SVM_CP_delete        = [[ [] for i in range(n_alpha_list)] for N in N_list]  #delete the constraint not in the dual solution
	times_SVM_CP_no_delete     = [[ [] for i in range(n_alpha_list)] for N in N_list]
	

	size_first_support = [[ [] for i in range(2)] for N in N_list] #get a sense of first support to adjust n_features
	n_samples          = 50

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
			alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)] #start with non empty support
			#alpha_list   = [1e-4*alpha_max]

			aux_alpha = -1


			index_SVM_CP_delete, _    = init_CP_norm_samples(X_train, y_train, n_samples, f) #just for highest alpha
			index_SVM_CP_no_delete, _ = init_CP_norm_samples(X_train, y_train, n_samples, f)


			model_L1_SVM           = 0
			model_SVM_CP_delete    = 0
			model_SVM_CP_no_delete = 0


			for alpha in alpha_list[::-1]:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)
				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM without CP #####', f)
				index_CP = init_CP_dual(X_train, y_train, alpha, n_samples, f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, _ = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM, False, f)
				times_L1_SVM[aux_N][aux_alpha].append(time_L1_SVM)



			#---L1 SVM with cutting planes and deleting
				write_and_print('\n###### L1 SVM with CP and deletion #####', f)
				beta_SVM_CP, support_SVM_CP, time_SVM_CP, model_SVM_CP_delete, index_SVM_CP_delete, _ = L1_SVM_CP(X_train, y_train, index_SVM_CP_delete, alpha, epsilon_RC, time_limit, model_SVM_CP_delete, True, f)
				times_SVM_CP_delete[aux_N][aux_alpha].append(time_SVM_CP)
				print len(index_SVM_CP_delete)


			#---L1 SVM with cutting planes and non deleting
				write_and_print('\n###### L1 SVM with CP and no deletion #####', f)
				beta_SVM_CP, support_SVM_CP, time_SVM_CP, model_SVM_CP_no_delete, index_SVM_CP_no_delete, _ = L1_SVM_CP(X_train, y_train, index_SVM_CP_no_delete, alpha, epsilon_RC, time_limit, model_SVM_CP_no_delete, False, f)
				times_SVM_CP_no_delete[aux_N][aux_alpha].append(time_SVM_CP)
				print len(index_SVM_CP_no_delete)



				same_support = len(set(support_SVM_CP)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CP))==0
				same_supports[aux_N][aux_alpha].append(same_support)

				

	write_and_print('\nSame support:       '+str(same_supports), f)



#---Compare times
	times_L1_SVM_averaged              = [ [ np.sum([times_L1_SVM[N][i][loop]              for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_SVM_CP_averaged_delete       = [ [ np.sum([times_SVM_CP_delete[N][i][loop]       for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]
	times_SVM_CP_averaged_no_delete    = [ [ np.sum([times_SVM_CP_no_delete[N][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for N in range(len(N_list))]

	L1_SVM_CP_plots_path(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM_averaged, times_SVM_CP_averaged_delete, times_SVM_CP_averaged_no_delete)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()



