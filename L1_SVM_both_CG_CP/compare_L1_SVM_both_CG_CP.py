import numpy as np
import datetime
import os
import sys

from gurobipy import *

from L1_SVM_both_CG_CP_plots import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *
from R_L1_SVM import *

sys.path.append('../L1_SVM_CP')
from L1_SVM_CP import *
from init_L1_SVM_CP import *




def compare_L1_SVM_both_CG_CP(type_Sigma, N_P_list, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../L1_SVM_both_CG_CP_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS
	epsilon_RC  = 1e-2
	loop_repeat = 5

	alpha       = 1
	time_limit  = 60  


	times_L1_SVM               = [[] for N,P in N_P_list]
	times_penalizedSVM_R_SVM   = [[] for N,P in N_P_list]
	times_SAM_R_SVM            = [[] for N,P in N_P_list]
	times_SVM_CG_CP  		   = [[] for N,P in N_P_list]


	same_supports    = [[] for N,P in N_P_list]
	compare_norms    = [[] for N,P in N_P_list]


	aux=-1
	write_and_print('\n###### alpha='+str(alpha)+' #####\n', f)
	for N,P in N_P_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+' and P='+str(P)+'--------------------------------------', f)

		aux += 1
		for i in range(loop_repeat):
			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)


		#---L1 SVM 
			write_and_print('\n###### L1 SVM#####', f)
			beta_L1_SVM, support_L1_SVM, time_L1_SVM, index_CP = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, f)
			times_L1_SVM[aux].append(time_L1_SVM)


		#---R penalizedSVM L1 SVM 
			write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
			beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
			times_penalizedSVM_R_SVM[aux].append(time_R_L1_SVM)



		#---R SAM L1 SVM 
			write_and_print('\n###### L1 SVM with R: SAM #####', f)
			beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
			times_SAM_R_SVM[aux].append(time_R_L1_SVM)




		#---L1 SVM with cutting planes
			n_samples  = np.max([100, N/16, P/16])    #run 4 times
			n_features = n_samples                    #run 4 times


			write_and_print('\n###### L1 SVM with CG and CP#####', f)


		#---1/ reduce number samples
			index_CP        = init_CP_dual(X_train, y_train, alpha, n_samples, f)
			X_train_reduced = np.array([X_train[i,:] for i in index_CP])
			y_train_reduced = np.array([y_train[i]   for i in index_CP])


		#---2/ use CG and reduce now the columns
			index_SVM_CG = init_correlation(X_train_reduced, y_train_reduced, n_features, f)
			beta_SVM_CG, support_SVM_CG, time_SVM_CG_0, model_SVM_CG, index_SVM_CG = L1_SVM_CG(X_train_reduced, y_train_reduced, index_SVM_CG, alpha, epsilon_RC, time_limit, 0, f)

			X_train_reduced = np.array([X_train[:,j] for j in support_SVM_CG]).T


		#---3/ use CP and reduce now the samples
			beta_SVM_CP, support_SVM_CP, time_SVM_CP, index_CP = L1_SVM_CP(X_train_reduced, y_train, index_CP, alpha, epsilon_RC, time_limit, f)

			X_train_reduced = np.array([X_train[i,:] for i in index_CP])
			y_train_reduced = np.array([y_train[i]   for i in index_CP])


		#---4/ use CG and return the final estimator
			index_SVM_CG = init_correlation(X_train_reduced, y_train_reduced, n_features, f)
			beta_SVM_CG, support_SVM_CG, time_SVM_CG, model_SVM_CG, index_SVM_CG = L1_SVM_CG(X_train_reduced, y_train_reduced, index_SVM_CG, alpha, epsilon_RC, time_limit, 0, f)
			support = np.where(beta_SVM_CG != 0)[0]
			#write_and_print(+str(support), f)

			

		#---5/ Store
			times_SVM_CG_CP[aux].append(time_SVM_CG_0 + time_SVM_CP + time_SVM_CG)

			same_support = len(set(support_SVM_CG)-set(support_R_L1_SVM))==0 and len(set(support_R_L1_SVM)-set(support_SVM_CG))==0
			same_supports[aux].append(same_support)

			xi = np.array([1 - y_train[i]*np.dot(X_train[i,:], beta_L1_SVM) for i in index_CP])
			write_and_print('\nLen of xi>0: '+str(len(xi[xi>0])), f)

			write_and_print('\n#### Total time: '+str(time_SVM_CG_0 + time_SVM_CP + time_SVM_CG)+'\n', f)



	write_and_print('\nSame support: '+str(same_supports), f)

	L1_SVM_both_CG_CP_plots(type_Sigma, N_P_list, k0, rho, tau_SNR, times_L1_SVM, times_penalizedSVM_R_SVM, times_SAM_R_SVM, times_SVM_CG_CP)
	plt.savefig(pathname+'/compare_times.png')
	plt.close()




