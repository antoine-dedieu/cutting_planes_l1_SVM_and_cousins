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

	alpha       = 1e-1
	time_limit  = 30  


	times_L1_SVM               = [[] for N in N_list]
	times_penalizedSVM_R_SVM   = [[] for N in N_list]
	times_SAM_R_SVM            = [[] for N in N_list]
	times_SVM_CP               = [[] for N in N_list]

	same_supports = [[] for N in N_list]
	compare_norms = [[] for N in N_list]

	aux=-1
	for N in N_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+'--------------------------------------', f)

		aux += 1
		for i in range(loop_repeat):
			write_and_print('\n\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)


		#---L1 SVM 
			write_and_print('\n###### L1 SVM without CP #####', f)
			n_samples = max(N/8, 8*P) 
			index_CP = init_CP_dual(X_train, y_train, alpha, n_samples, f)
			beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _ = L1_SVM_CP(X_train, y_train, index_CP, alpha, epsilon_RC, time_limit, 0, f)
			times_L1_SVM[aux].append(time_L1_SVM)


		#---R penalizedSVM L1 SVM 
			#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
			#if N<1000:
			#	beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
			#else:
			#	time_R_L1_SVM = time_L1_SVM
			#times_penalizedSVM_R_SVM[aux].append(time_R_L1_SVM)



		#---R SAM L1 SVM 
			#write_and_print('\n###### L1 SVM with R: SAM #####', f)
			#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
			#times_SAM_R_SVM[aux].append(time_R_L1_SVM)



		#---L1 SVM with cutting planes
			n_samples = max(N/8, 8*P)    #run 3 times

			write_and_print('\n###### L1 SVM with CP #####', f)
			#index_CP = init_CP_dual(X_train, y_train, alpha, n_samples, f)
			index_CP = init_CP_norm_samples(X_train, y_train, n_samples, f)
			beta_SVM_CP, support_SVM_CP, time_SVM_CP, _, _ = L1_SVM_CP(X_train, y_train, index_CP, alpha, epsilon_RC, time_limit, 0, f)


			times_SVM_CP[aux].append(time_SVM_CP)

			same_support = len(set(support_SVM_CP)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CP))==0
			same_supports[aux].append(same_support)

			xi = np.array([1 - y_train[i]*np.dot(X_train[i,:], beta_L1_SVM) for i in index_CP])
			write_and_print('\nLen of xi>0: '+str(len(xi[xi>0])), f)



	write_and_print('\nSame support: '+str(same_supports), f)

	L1_SVM_CP_plots(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM, times_penalizedSVM_R_SVM, times_SAM_R_SVM, times_SVM_CP)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()

	#return times_L1_SVM, times_SVM_CP, same_supports



