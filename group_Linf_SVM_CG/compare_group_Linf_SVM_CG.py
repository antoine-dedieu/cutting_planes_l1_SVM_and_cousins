import numpy as np
import datetime
import time
import os
import sys
import math
import subprocess


from gurobipy import *
from group_Linf_SVM_CG import *
from group_Linf_SVM_CG_plots import *
from simulate_data_group import *
from smoothing_proximal_group_Linf import *

#from benchmark import *


sys.path.append('../algortihms')
from smoothing_hinge_loss import *








def compare_group_Linf_SVM_CG(type_Sigma, N, P_list, k0, rho, tau_SNR):

#N_list: list of values to average


	DT = datetime.datetime.now()
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-N'+str(N)+'_k0'+str(k0)+'_rho'+str(rho)+'_tau'+str(tau_SNR)+'_Sigma'+str(type_Sigma)
	pathname=r'../../../group_Linf_SVM_CG_results/'+str(name)
	os.makedirs(pathname)

	#FILE RESULTS
	f = open(pathname+'/results.txt', 'w')


#---PARAMETERS
	epsilon_RC  = 1e-2
	loop_repeat = 3
	
	n_alpha_list = 1
	time_limit   = 30  


	## Feature for each group
	feat_by_group = 10



#---RESULTS
	times_group_Linf_SVM       = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_2_bis  = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_3      = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_3_bis  = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_4      = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	times_SVM_CG_method_4_bis  = [[ [] for i in range(n_alpha_list)] for P in P_list]  


	objvals_group_Linf_SVM     = [[ [] for i in range(n_alpha_list)] for P in P_list]
	objvals_SVM_method_1       = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	objvals_SVM_method_2       = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	objvals_SVM_method_3       = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	objvals_SVM_method_4       = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	


	aux_P = -1
	for P in P_list:

		write_and_print('\n\n\n-----------------------------For P='+str(P)+'--------------------------------------', f)

		aux_P += 1
		for i in range(loop_repeat):

			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)


		#---Simulate data
			seed_X        = random.randint(0,1000)
			group_to_feat = np.array([range(10*k, 10*(k+1)) for k in range(P/feat_by_group)]) 
			X_train, l2_X_train, y_train, u_positive = simulate_data_group(type_Sigma, N, P, group_to_feat, k0, rho, tau_SNR, seed_X, f)



		#---Lambda max
			aux          = np.sum( np.abs(X_train), axis=0)
			aux          = [np.sum(aux[idx]) for idx in group_to_feat]

			alpha_max    = np.max(aux)                
			alpha_list   = [.1*alpha_max]



		#---Initialization
			n_groups = N/2
			idx_groups_CG, time_correl = init_group_Linf(X_train, y_train, group_to_feat, n_groups, f)


			model_group_Linf_SVM  = 0
			model_method_1        = 0
			model_method_2        = 0
			model_method_3        = 0
			model_method_4        = 0


		#---Nesterov
			abs_correlations  = np.abs(np.dot(X_train.T, y_train) )
			sum_correl_groups = [np.sum(abs_correlations[idx]) for idx in group_to_feat]

			N_groups          = N
			index_groups      = np.argsort(sum_correl_groups)[::-1][:N_groups]


			X_train_reduced       = np.zeros((N,0))
			group_to_feat_reduced = []
			aux = 0


		#--Order in groups
			for i in range(N_groups):
				X_train_reduced = np.concatenate([X_train_reduced, np.array([X_train[:,j] for j in group_to_feat[index_groups[i]]]).T ], axis=1)
				group_to_feat_reduced.append(range(aux, aux+len(group_to_feat[index_groups[i]]) ))
				aux += len(group_to_feat[index_groups[i]]) 


			write_and_print('\n\n###### Nesterov method without CD #####', f)
			tau_max = .2
			n_loop  = 1
			n_iter  = 50
			index_groups_method_2, time_smoothing_2, beta_method_2 = loop_smoothing_proximal_group_Linf('hinge', 'l1_linf', X_train_reduced, y_train, group_to_feat_reduced, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			index_groups_method_2 = np.array(index_groups)[index_groups_method_2].tolist()


			write_and_print('\n\n###### Nesterov method with CD #####', f)
			tau_max = .2
			n_loop  = 1
			n_iter  = 50
			index_groups_method_3, time_smoothing_3, beta_method_3 = loop_smoothing_proximal_group_Linf('hinge', 'l1_linf_CD', X_train_reduced, y_train, group_to_feat_reduced, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			index_groups_method_3 = np.array(index_groups)[index_groups_method_3].tolist()


			#write_and_print('\n\n###### Nesterov method with squared hinge and CD #####', f)
			#tau_max = .2
			#n_loop  = 1
			#n_iter  = 50
			#index_groups_method_4, time_smoothing_4, beta_method_4 = loop_smoothing_proximal_group_Linf('squared_hinge', 'l1_linf_CD', X_train_reduced, y_train, group_to_feat_reduced, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			#index_groups_method_4 = np.array(index_groups)[index_groups_method_4].tolist()




			aux_alpha = -1
			for alpha in alpha_list:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)
				aux_alpha += 1

			#---group_Linf SVM 
				write_and_print('\n###### group_Linf SVM with Gurobi without CG #####', f)
				if P<1e2:
					beta_group_Linf_SVM, support_group_Linf_SVM, time_group_Linf_SVM, model_group_Linf_SVM, _, obj_val_group_Linf_SVM = group_Linf_SVM_CG(X_train, y_train, group_to_feat, range(group_to_feat.shape[0]), alpha_list[0], 0, time_limit, model_group_Linf_SVM, [], False, f) #_ = range(P) 
				else:
					time_group_Linf_SVM = 1
					obj_val_group_Linf_SVM = 1

				times_group_Linf_SVM[aux_P][aux_alpha].append(time_group_Linf_SVM)


			#---Regularization path
				write_and_print('\n###### group_Linf SVM path with CG correl 5 groups #####', f)
				alpha_bis = alpha_max
				beta_method_1     = []
				time_method_1_tot = 0

				while 0.5*alpha_bis > alpha_list[0]:
					beta_method_1, support_method_1, time_method_1, model_method_1, idx_groups_CG, obj_val_method_1   = group_Linf_SVM_CG(X_train, y_train, group_to_feat, idx_groups_CG, alpha_bis, 1e-2, time_limit, model_method_1, beta_method_1, False, f)
					alpha_bis   *= 0.5
					time_method_1_tot += time_method_1
				beta_method_1, support_method_1, time_method_1, model_method_1, idx_groups_CG, obj_val_method_1   = group_Linf_SVM_CG(X_train, y_train, group_to_feat, idx_groups_CG, alpha_list[0], 1e-2, time_limit, model_method_1, beta_method_1, False, f)
				time_method_1_tot += time_method_1

				times_SVM_CG_method_1[aux_P][aux_alpha].append(time_method_1_tot)
				objvals_SVM_method_1[aux_P][aux_alpha].append(obj_val_method_1/float(obj_val_group_Linf_SVM +1e-5) )

				write_and_print('\nTIME ALL CG = '+str(time_method_1_tot), f)


			#---Nesterov path
				write_and_print('\n\n###### group_Linf SVM with CG + AGD, without CD #####', f)
				beta_method_2, support_method_2, time_method_2, model_method_2, index_columns_method_2, obj_val_method_2 = group_Linf_SVM_CG(X_train, y_train, group_to_feat, index_groups_method_2, alpha_list[0], 1e-2, time_limit, model_method_2, [], False, f)
				times_SVM_CG_method_2[aux_P][aux_alpha].append(time_correl + time_smoothing_2 + time_method_2)
				objvals_SVM_method_2[aux_P][aux_alpha].append(obj_val_method_2/float(obj_val_group_Linf_SVM +1e-5) )

				times_SVM_CG_method_2_bis[aux_P][aux_alpha].append(time_method_2)



			#---Nesterov path Block Coordinate
				write_and_print('\n\n###### group_Linf SVM with CG, with block CD  #####', f)
				beta_method_3, support_method_3, time_method_3, model_method_3, index_columns_method_3, obj_val_method_3 = group_Linf_SVM_CG(X_train, y_train, group_to_feat, index_groups_method_3, alpha_list[0], 1e-2, time_limit, model_method_3, [], False, f)
				times_SVM_CG_method_3[aux_P][aux_alpha].append(time_correl + time_smoothing_3 + time_method_3)
				objvals_SVM_method_3[aux_P][aux_alpha].append(obj_val_method_3/float(obj_val_group_Linf_SVM +1e-5) )

				times_SVM_CG_method_3_bis[aux_P][aux_alpha].append(time_method_3)



			#---Nesterov path Block Coordinate and Squared Hinge
				#write_and_print('\n\n###### group_Linf SVM with CG, with squared hinge  block CD  #####', f)
				#beta_method_4, support_method_4, time_method_4, model_method_4, index_columns_method_4, obj_val_method_4 = group_Linf_SVM_CG(X_train, y_train, group_to_feat, index_groups_method_4, alpha_list[0], 1e-2, time_limit, model_method_4, [], False, f)
				#times_SVM_CG_method_4[aux_P][aux_alpha].append(time_correl + time_smoothing_4 + time_method_4)
				#objvals_SVM_method_4[aux_P][aux_alpha].append(obj_val_method_4/float(obj_val_group_Linf_SVM +1e-5) )

				#times_SVM_CG_method_4_bis[aux_P][aux_alpha].append(time_method_3)

	



#---Compare times
	times_group_Linf_SVM  = [ [ np.sum([times_group_Linf_SVM[P][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_1        = [ [ np.sum([times_SVM_CG_method_1[P][i][loop]     for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_2        = [ [ np.sum([times_SVM_CG_method_2[P][i][loop]     for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_2_bis    = [ [ np.sum([times_SVM_CG_method_2_bis[P][i][loop] for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	
	times_method_3        = [ [ np.sum([times_SVM_CG_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_3_bis    = [ [ np.sum([times_SVM_CG_method_3_bis[P][i][loop] for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	
	
	#times_benchmark = [ [ np.sum([times_benchmark[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]


	#legend_plot_1 = {0:'RP path', 1:'Correlation 5N + T_max=100', 2:'Correlation 10N + T_max=100', 3:'Correl top 50'}
	legend_plot_1 = {0:'Gurobi + CG', 1:'Nesterov + CG', 2:'Nesterov given, CG', 3:'Nesterov block CD + CG', 4:'Nesterov block CD given, CG'}


#####################

	np.save(pathname+'/times_group_Linf_SVM', times_group_Linf_SVM)
	np.save(pathname+'/times_method_1', times_method_1)
	np.save(pathname+'/times_method_2', times_method_2)
	np.save(pathname+'/times_method_2_bis', times_method_2_bis)
	np.save(pathname+'/times_method_3', times_method_3)
	np.save(pathname+'/times_method_3_bis', times_method_3_bis)

	
	times_list    = [times_method_1, times_method_2, times_method_2_bis, times_method_3, times_method_3_bis]
	P_list        = [str(int(0.001*P))+'K' for P in P_list]

	group_Linf_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_large.pdf', bbox_inches='tight')

	group_Linf_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_small.pdf', bbox_inches='tight')
	plt.close()





#---Compare objective values
	objvals_SVM_CG_method_0    = [ [1 for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_1    = [ [ np.mean([objvals_SVM_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_2    = [ [ np.mean([objvals_SVM_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]


	np.save(pathname+'/objvals_SVM_CG_method_0', objvals_SVM_CG_method_0)
	np.save(pathname+'/objvals_SVM_CG_method_1', objvals_SVM_CG_method_1)
	np.save(pathname+'/objvals_SVM_CG_method_2', objvals_SVM_CG_method_2)
	

	objvals_list    = [objvals_SVM_CG_method_0, objvals_SVM_CG_method_1, objvals_SVM_CG_method_2, objvals_SVM_CG_method_2]

	group_Linf_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_1, 'objval','large')
	plt.savefig(pathname+'/compare_objective_values_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_objective_values_large.pdf', bbox_inches='tight')
	plt.close()

	group_Linf_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_1, 'objval','small')
	plt.savefig(pathname+'/compare_objective_values_small.pdf', bbox_inches='tight')
	plt.close()






