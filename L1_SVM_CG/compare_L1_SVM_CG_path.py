import numpy as np
import datetime
import time
import os
import sys


from gurobipy import *
from L1_SVM_CG import *
from L1_SVM_CG_plots import *
from benchmark import *

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
	epsilon_RC  = 1e-2
	loop_repeat = 5
	
	n_alpha_list = 20
	time_limit   = 30  



#---RESULTS
	times_L1_SVM_naive         = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_L1_SVM_no_WS         = [[ [] for i in range(n_alpha_list)] for P in P_list]

	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_method_3      = [[ [] for i in range(n_alpha_list)] for P in P_list]

	objvals_SVM_method_1       = [[ [] for i in range(n_alpha_list)] for P in P_list]
	objvals_SVM_method_2       = [[ [] for i in range(n_alpha_list)] for P in P_list] 
	objvals_SVM_method_3       = [[ [] for i in range(n_alpha_list)] for P in P_list]
	
	n_features = 10


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
			#alpha_list   = [1e-1*alpha_max]

			aux_alpha = -1


			index_method_1,_    = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
			index_method_2,_    = init_correlation(X_train, y_train, n_features, f)
			index_method_3,_    = init_correlation(X_train, y_train, n_features, f)


			model_L1_SVM_no_WS= 0
			model_L1_SVM_WS   = 0
			model_method_1    = 0
			model_method_2    = 0
			model_method_3    = 0
			




			for alpha in alpha_list:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM with Gurobi without keeping model #####', f)
				if P<5e4:
					beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, 0, [], False, f) #_ = range(P) 
				else:
					time_L1_SVM = 1
				times_L1_SVM_naive[aux_P][aux_alpha].append(time_L1_SVM)

			#---L1 SVM 
				write_and_print('\n###### L1 SVM with Gurobi without CG #####', f)
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM_WS, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM_WS, [], False, f) #_ = range(P) 
				times_L1_SVM[aux_P][aux_alpha].append(time_L1_SVM)


			#---L1 SVM DEBILE
				#beta_L1_SVM_WS    = np.zeros(P) if aux_alpha == 0 else []
				#write_and_print('\n###### L1 SVM with Gurobi without CG without keeping model #####', f)
				#beta_L1_SVM_WS, support_L1_SVM, time_L1_SVM, _, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, 0, beta_L1_SVM_WS, False, f) #_ = range(P) 
				#times_L1_SVM_no_WS[aux_P][aux_alpha].append(time_L1_SVM)


			#---R penalizedSVM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_penalizedSVM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)

			#---R SAM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: SAM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_SAM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)



			#---L1 SVM with CG 
				beta_method_1    = np.zeros(P) if aux_alpha == 0 else []
				write_and_print('\n###### L1 SVM with CG correlation, eps=5e-1 #####', f)
				beta_method_1, support_method_1, time_method_1, model_method_1, index_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_method_1, alpha, 5e-1, time_limit, model_method_1, beta_method_1, False, f)
				times_SVM_CG_method_1[aux_P][aux_alpha].append(time_method_1)
				objvals_SVM_method_1[aux_P][aux_alpha].append(obj_val_method_1/float(obj_val_L1_SVM))


			#---L1 SVM with CG 
				beta_method_2    = np.zeros(P) if aux_alpha == 0 else []
				write_and_print('\n###### L1 SVM with CG correlation, eps=1e-1 #####', f)
				beta_method_2, support_method_2, time_method_2, model_method_2, index_method_2, obj_val_method_2  = L1_SVM_CG(X_train, y_train, index_method_2, alpha, 1e-1, time_limit, model_method_2, beta_method_2, False, f)
				times_SVM_CG_method_2[aux_P][aux_alpha].append(time_method_2)
				objvals_SVM_method_2[aux_P][aux_alpha].append(obj_val_method_2/float(obj_val_L1_SVM))

			#---L1 SVM with CG 
				beta_method_3    = np.zeros(P) if aux_alpha == 0 else []
				write_and_print('\n###### L1 SVM with CG correlation, eps=1e-2 #####', f)
				beta_method_3, support_method_3, time_method_3, model_method_3, index_method_3, obj_val_method_3  = L1_SVM_CG(X_train, y_train, index_method_3, alpha, 1e-2, time_limit, model_method_3, beta_method_3, False, f)
				times_SVM_CG_method_3[aux_P][aux_alpha].append(time_method_3)
				objvals_SVM_method_3[aux_P][aux_alpha].append(obj_val_method_3/float(obj_val_L1_SVM))




				#same_support = len(set(support_SVM_CG)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CG))==0
				#same_supports[aux_P][aux_alpha].append(same_support)
				#compare_norms[aux_P][aux_alpha].append(np.linalg.norm(beta_SVM_CG - beta_L1_SVM)**2)
				#compare_ratio[aux_P][aux_alpha].append(time_L1_SVM/time_SVM_CG)


			#---Get a sense of size of first support to adjust n
				#if aux_alpha<2:
				#	size_first_support[aux_P][aux_alpha].append(len(support_SVM_CG))


			#support = np.array(support)
			#print [set(support[i]) < set(support[i+1]) for i in range(len(support) -1)]



	#write_and_print('\nSame support:       '+str(same_supports), f)
	#write_and_print('\nSize first support: '+str(size_first_support), f)


#---Ratio of win
	#compare_ratio = np.array([ [ np.mean([times_L1_SVM[P][i][loop] for loop in range(loop_repeat) ]) for i in range(n_alpha_list)] for P in range(len(P_list))])
	#compare_ratios(type_Sigma, N, P_list, k0, rho, tau_SNR, compare_ratio, alpha_list)
	#plt.savefig(pathname+'/compare_ratios.png')
	#plt.close()

	



#---Compare times
	times_L1_SVM_naive   = [ [ np.sum([times_L1_SVM_naive[P][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	
	times_L1_SVM   = [ [ np.sum([times_L1_SVM[P][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	#times_L1_SVM_no_WS= [ [ np.sum([times_L1_SVM_no_WS[P][i][loop]          for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_1 = [ [ np.sum([times_SVM_CG_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_2 = [ [ np.sum([times_SVM_CG_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_3 = [ [ np.sum([times_SVM_CG_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]




#---PLOT 1 Compare everybody
	legend_plot_1 = {0:'Naive', 1:'No CG', 2:'CG Eps=5e-1', 3:'CG Eps=1e-1', 4:'CG Eps=1e-2'}
	times_list  = [times_L1_SVM_naive, times_L1_SVM, times_method_1, times_method_2, times_method_3]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time', 'large')
	plt.savefig(pathname+'/compare_times_errorbar.pdf')
	plt.close()


#---STORE tables
	times = open(pathname+'/times.txt', 'w')
	write_and_print('P list: '+str(P_list), times)

	for i in range(len(times_list)):
		array = np.array(times_list)[i]
		mean_method = np.mean(array, axis=1) 
		std_method  = np.std(array,  axis=1)

		write_and_print(str(legend_plot_1[i])+ ' :', times)
		write_and_print('Mean: '+ str(mean_method ) +' Std:'+ str(std_method) +'\n', times)


#---PLOT 2 Compare subgroup
	legend_plot_2 = {0:'CG Eps=5e-1', 1:'CG Eps=1e-1', 2:'CG Eps=1e-2'}
	times_list  = [times_method_1, times_method_2, times_method_3]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup.pdf')
	plt.close()








#---Compare objective values
	objvals_SVM_CG_method_0    = [ [1 for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_1    = [ [ np.mean([objvals_SVM_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_2    = [ [ np.mean([objvals_SVM_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_3    = [ [ np.mean([objvals_SVM_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]



#---PLOT 2 Compare subgroup
	objvals_list  = [objvals_SVM_CG_method_1, objvals_SVM_CG_method_2, objvals_SVM_CG_method_3]
	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_2, 'objval','large')
	plt.savefig(pathname+'/compare_objective_values_subgroup.pdf')
	plt.close()


#---STORE tables
	objvals = open(pathname+'/objvals.txt', 'w')
	write_and_print('P list: '+str(P_list), objvals)

	for i in range(len(objvals_list)):
		mean_method = np.round(np.mean(np.array(objvals_list)[i], axis=1), 4)
		std_method  = np.round(np.std(np.array(objvals_list)[i],  axis=1), 4)
		write_and_print(str(legend_plot_2[i])+ ' :\n', objvals)
		write_and_print('Mean: '+ str(mean_method) +' Std:'+ str(std_method) +'\n\n', objvals)




