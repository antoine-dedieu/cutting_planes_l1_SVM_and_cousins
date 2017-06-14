import numpy as np
import datetime
import time
import os
import sys
import math
import subprocess


from gurobipy import *
from L1_SVM_CG import *
from L1_SVM_CG_plots import *

from benchmark import *
from R_L1_SVM import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../algortihms')
from smoothing_hinge_loss import *








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
	
	n_alpha_list = 1
	time_limit   = 30  




#---RESULTS
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_L1_SVM_WS            = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for P in P_list]  #delete the columns not in the support
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_method_3      = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_SVM_CG_method_4      = [[ [] for i in range(n_alpha_list)] for P in P_list]
	times_benchmark            = [[ [] for i in range(n_alpha_list)] for P in P_list]

	objvals_L1_SVM             = [[ [] for i in range(n_alpha_list)] for P in P_list]
	objvals_SVM_method_1       = [[ [] for i in range(n_alpha_list)] for P in P_list]
	objvals_SVM_method_2       = [[ [] for i in range(n_alpha_list)] for P in P_list] 
	objvals_SVM_method_3       = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	objvals_SVM_method_4       = [[ [] for i in range(n_alpha_list)] for P in P_list]
	objvals_benchmark          = [[ [] for i in range(n_alpha_list)] for P in P_list]  
	


	aux_P = -1
	for P in P_list:

		write_and_print('\n\n\n-----------------------------For P='+str(P)+'--------------------------------------', f)

		aux_P += 1
		for i in range(loop_repeat):

			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)


		#---Simulate data
			seed_X = random.randint(0,1000)
			X_train, X_test, l2_X_train, y_train, y_test, u_positive = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

			for j in range(P):    
				X_train[:,j] *= l2_X_train[j]


			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
			alpha_list   = [1e-6*alpha_max]


		#---STORE FOR COMPARISONS
			#store_ADMM_SVM_comparison(X_train, y_train, N, P, seed_X, 1e-2*alpha_max)



			n_features = 50
			index_SVM_CG_correl, time_correl   = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
			index_SVM_CG_liblinear, time_liblinear, beta_liblinear = liblinear_for_CG('squared_hinge_l1', X_train, y_train, alpha_list[0], f)



		#---METHOD 2
			#scaling = (200./N)**2

			argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
			index_CG        = argsort_columns[::-1][:5*N]
			X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T

			tau_max = 0.1
			n_loop  = 1
			n_iter  = 5

			#tau_max = 1*scaling
			tau_max = 1
			n_loop  = 1
			n_iter  = 50

			write_and_print('\n\n###### Method 2 #####', f)
			index_samples_method_2, index_columns_method_2, time_smoothing_2, beta_method_2 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			index_columns_method_2 = np.array(index_CG)[index_columns_method_2].tolist()
			
			#if len(index_columns_method_2) >N:
			#	argsort_columns         = np.argsort(np.abs(beta_method_2))
			#	index_columns_method_2  = argsort_columns[::-1][:N].tolist()


		#---METHOD 3
			argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
			index_CG        = argsort_columns[::-1][:10*N]
			X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T

			#tau_max = 1*scaling
			tau_max = 0.2
			n_loop  = 1
			n_iter  = 200

			write_and_print('\n\n###### Method 3 #####', f)

			index_samples_method_3, index_columns_method_3, time_smoothing_3, beta_method_3 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			index_columns_method_3 = np.array(index_CG)[index_columns_method_3].tolist()

			#if len(index_columns_method_3) >N:
			#	argsort_columns         = np.argsort(np.abs(beta_method_3))
			#	index_columns_method_3  = argsort_columns[::-1][:N].tolist()

			#index_columns_method_3 = np.array(index_CG)[index_columns_method_3_bis].tolist()
			#index_columns_method_3 = index_columns_method_3_bis

			#n_loop  = 20
			#n_iter  = 10
			#X_train_reduced_features = np.array([X_train[:,j] for j in index_columns_method_3]).T
			#index_samples_method_3, index_columns_method_3_bis, time_smoothing, beta_smoothing_bis = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced_features, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			#index_columns_method_3 = np.array(index_CG)[index_columns_method_3_bis].tolist()

			#beta_method_3 = np.zeros(P)
			#for j in range(len(index_columns_method_3_bis)):
			#    beta_method_3[index_columns_method_3_bis[j]] = beta_smoothing_bis[j]



		#---METHOD 4
			#write_and_print('\n\n###### Method 4 #####', f)
			#index_samples_method_4, index_columns_method_4, time_smoothing_4, beta_method_4 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
			#index_columns_method_4 = np.array(index_CG)[index_columns_method_4].tolist()
			#argsort_coeffs         = np.argsort(np.abs(beta_method_4))
			#index_columns_method_4 = argsort_coeffs[::-1][:100].tolist()



			model_L1_SVM      = 0
			model_method_1    = 0
			model_method_2    = 0
			model_method_3    = 0
			model_method_4    = 0



			aux_alpha = -1
			for alpha in alpha_list:
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

				aux_alpha += 1

			#---L1 SVM 
				write_and_print('\n###### L1 SVM with Gurobi without CG without warm start #####', f)
				if P<1e6:
					beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha_list[0], 0, time_limit, model_L1_SVM, [], False, f) #_ = range(P) 
				else:
					#beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM, [], False, f) #_ = range(P) 
					#beta_L1_SVM, _, time_L1_SVM = SVM_ADMM(X_train, y_train, alpha, f)
					time_L1_SVM = 1
					obj_val_L1_SVM = 1
					
				times_L1_SVM[aux_P][aux_alpha].append(time_L1_SVM)


			#---Compute path
				if P<1e6:
					write_and_print('\n###### L1 SVM path with CG correl 10 #####', f)
					alpha_bis = alpha_max
					index_columns_method_1, _   = init_correlation(X_train, y_train, 10, f)
					beta_method_1     = []
					time_method_1_tot = 0

					while 0.5*alpha_bis > alpha_list[0]:
						beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_bis, 1e-2, time_limit, model_method_1, beta_method_1, False, f)
						alpha_bis   *= 0.5
						time_method_1_tot += time_method_1
					beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_list[0], 1e-2, time_limit, model_method_1, beta_method_1, False, f)
					time_method_1_tot += time_method_1

					times_SVM_CG_method_1[aux_P][aux_alpha].append(time_method_1_tot)
					objvals_SVM_method_1[aux_P][aux_alpha].append(obj_val_method_1/float(obj_val_L1_SVM) -1.)

					write_and_print('\nTIME ALL CG = '+str(time_method_1_tot), f)



				#print '\nNorm diff: '+str(np.linalg.norm(beta_L1_SVM - beta_smoothing[:P]))


			#---L1 SVM 
				#write_and_print('\n###### L1 SVM with Gurobi without CG with warm start #####', f)
				#beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM_WS, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM_WS, beta_liblinear, False, f) #_ = range(P) 
				#times_L1_SVM_WS[aux_P][aux_alpha].append(time_L1_SVM)


			#---R penalizedSVM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_penalizedSVM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)

			#---R SAM L1 SVM 
				#write_and_print('\n###### L1 SVM with R: SAM #####', f)
				#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)
				#times_SAM_R_SVM[aux_P][aux_alpha].append(time_R_L1_SVM)



			#---L1 SVM with CG and deleting
				#obj_val_L1_SVM = 1
				#write_and_print('\n###### L1 SVM with CG correlation, eps=5e-2 #####', f)
				#write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 1k, tau='+str(scaling)+', T_max = 50  #####', f)
				#beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha, 5e-2, time_limit, model_method_1, [], False, f)
				#beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_SVM_CG_correl, alpha, 5e-2, time_limit, model_method_1, [], False, f)
				
				#times_SVM_CG_method_1[aux_P][aux_alpha].append(time_correl + time_smoothing_1 + time_method_1)
				#objvals_SVM_method_1[aux_P][aux_alpha].append(obj_val_method_1/float(obj_val_L1_SVM))


			#---L1 SVM with CG and not deleting
				#write_and_print('\n###### L1 SVM with CG restricted, eps=5e-2 #####', f)
				write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 5n, tau='+str(1)+', T_max = 100 #####', f)
				#beta_method_2, support_method_2, time_method_2, model_method_2, index_columns_method_2, obj_val_method_2  = L1_SVM_CG(X_train, y_train, index_columns_method_2, alpha, 5e-2, time_limit, model_method_2, [], False, f)
				#times_SVM_CG_method_2[aux_P][aux_alpha].append(time_correl + time_smoothing_2 + time_method_2)
				#objvals_SVM_method_2[aux_P][aux_alpha].append(obj_val_method_2/float(obj_val_L1_SVM))


			#---L1 SVM with CG and not deleting
				write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 10n, tau='+str(1)+', T_max = 100 #####', f)
				beta_method_3, support_method_3, time_method_3, model_method_3, index_columns_method_3, obj_val_method_3 = L1_SVM_CG(X_train, y_train, index_columns_method_3, alpha_list[0], 1e-2, time_limit, model_method_3, [], False, f)
				times_SVM_CG_method_3[aux_P][aux_alpha].append(time_correl + time_smoothing_3 + time_method_3)
				objvals_SVM_method_3[aux_P][aux_alpha].append(obj_val_method_3/float(obj_val_L1_SVM) -1.)


			#---L1 SVM with correlation
				write_and_print('\n\n###### L1 SVM with CG correl 50 #####', f)
				beta_method_4, support_method_4, time_method_4, model_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_CG(X_train, y_train, index_SVM_CG_correl, alpha_list[0], 1e-2, time_limit, model_method_4, [], False, f)
				
				times_SVM_CG_method_4[aux_P][aux_alpha].append(time_correl + time_method_4)
				objvals_SVM_method_4[aux_P][aux_alpha].append(obj_val_method_4/float(obj_val_L1_SVM) -1.)

				times_SVM_CG_method_4[aux_P][aux_alpha].append(1)
				objvals_SVM_method_4[aux_P][aux_alpha].append(0)



			#---L1 SVM with CG and not deleting
				#write_and_print('\n\n###### L1 SVM with CG squared-hinge AGD, eps=5e-2  #####', f)
				#write_and_print(str(sorted(index_columns_method_4)), f)
				#beta_method_4, support_method_4, time_method_4, model_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_CG(X_train, y_train, index_columns_method_4, alpha, 5e-2, time_limit, model_method_4, [], False, f)
				#times_SVM_CG_method_4[aux_P][aux_alpha].append(time_method_4)
				#objvals_SVM_method_4[aux_P][aux_alpha].append(obj_val_method_4/float(obj_val_L1_SVM))



			#---BENCHMARK
				write_and_print('\n\n###### Benchmark AL_CD #####', f)
				store_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, seed_X, 1e-6*alpha_max, 'single')
				subprocess.call([pathname+'/../../best_subset_classification/LPsparse/LPsparse', '-d', pathname+'/../../best_subset_classification/LPsparse/data/synthetic_dataset/data_train_single'])
				obj_val_AL_CD, time_AL_CD = check_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, 1e-6*alpha_max, os.path.dirname(os.path.realpath(__file__)), f)
				times_benchmark[aux_P][aux_alpha].append(time_AL_CD)
				objvals_benchmark[aux_P][aux_alpha].append(obj_val_AL_CD/float(obj_val_L1_SVM) -1.)




	



#---Compare times
	times_L1_SVM    = [ [ np.sum([times_L1_SVM[P][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	#times_L1_SVM_WS= [ [ np.sum([times_L1_SVM_WS[P][i][loop]         for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_1  = [ [ np.sum([times_SVM_CG_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	#times_method_2 = [ [ np.sum([times_SVM_CG_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_3  = [ [ np.sum([times_SVM_CG_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_4  = [ [ np.sum([times_SVM_CG_method_4[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_benchmark = [ [ np.sum([times_benchmark[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]


	#legend_plot_1 = {0:'RP path', 1:'Correlation 5N + T_max=100', 2:'Correlation 10N + T_max=100', 3:'Correl top 50'}
	legend_plot_1 = {0:'RP CG', 1:'FO + CG', 2:'AL-CD', 3:'Correlation screening ', 4:'LP solver'}

	np.save(pathname+'/times_L1_SVM', times_L1_SVM)
	np.save(pathname+'/times_method_1', times_method_1)
	np.save(pathname+'/times_method_3', times_method_3)
	np.save(pathname+'/times_method_4', times_method_4)
	np.save(pathname+'/times_benchmark', times_benchmark)

	
	times_list    = [times_method_1, times_method_3, times_benchmark, times_method_4, times_L1_SVM]
	P_list        = [str(int(0.001*P))+'K' for P in P_list]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_large.pdf', bbox_inches='tight')

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_small.pdf', bbox_inches='tight')
	plt.close()


	#legend_plot_2 = {0:'Regularization path', 1:'Correlation 5N + T_max=100', 2:'Correlation 10N + T_max=100', 3:'Benchmark AL_CD'}
	legend_plot_2 = {0:'RP CG', 1:'FO + CG', 2:'AL-CD'}
	times_list    = [times_method_1, times_method_3, times_benchmark]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_subgroup_large.pdf', bbox_inches='tight')
	plt.close()

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_small.pdf', bbox_inches='tight')
	plt.close()




#---Compare objective values
	#objvals_SVM_CG_method_0    = [ [1 for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_1    = [ [ np.mean([objvals_SVM_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	#objvals_SVM_CG_method_2    = [ [ np.mean([objvals_SVM_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_3    = [ [ np.mean([objvals_SVM_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_SVM_CG_method_4    = [ [ np.mean([objvals_SVM_method_4[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	objvals_benchmark 		   = [ [ np.sum([objvals_benchmark[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]

	np.save(pathname+'/objvals_SVM_CG_method_1', objvals_SVM_CG_method_1)
	np.save(pathname+'/objvals_SVM_CG_method_3', objvals_SVM_CG_method_3)
	np.save(pathname+'/objvals_SVM_CG_method_4', objvals_SVM_CG_method_4)
	np.save(pathname+'/objvals_benchmark', objvals_benchmark)



	objvals_list    = [objvals_SVM_CG_method_1, objvals_SVM_CG_method_3, objvals_benchmark]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_2, 'objval','large')
	plt.savefig(pathname+'/compare_objective_values_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_objective_values_large.pdf', bbox_inches='tight')
	plt.close()

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_2, 'objval','small')
	plt.savefig(pathname+'/compare_objective_values_small.pdf', bbox_inches='tight')
	plt.close()

	#np.save(pathname+'/../../L1_SVM_results_to_plot/objvals_SVM_CG_method_0', times_L1_SVM)
	#np.save(pathname+'/../../L1_SVM_results_to_plot/objvals_SVM_CG_method_1', objvals_SVM_CG_method_1)
	#np.save(pathname+'/../../L1_SVM_results_to_plot/objvals_SVM_CG_method_2', objvals_SVM_CG_method_2)
	#np.save(pathname+'/../../L1_SVM_results_to_plot/objvals_SVM_CG_method_3', objvals_SVM_CG_method_4)









def plot():

	current_path = os.path.dirname(os.path.realpath(__file__))

	pathname = current_path+'/../../../L1_SVM_results_to_plot'

#---LOAD 
	times_method_0 = np.load(pathname+'/times_SVM_CG_method_0').reshape(5,1)
	times_method_1 = np.load(pathname+'/times_SVM_CG_method_1').reshape(5,1)
	times_method_2 = np.load(pathname+'/times_SVM_CG_method_2').reshape(5,1)
	times_method_3 = np.load(pathname+'/times_SVM_CG_method_3').reshape(5,1)



#---LOAD 
	objvals_method_0 = np.load(pathname+'/objvals_SVM_CG_method_0').reshape(5,1)
	objvals_method_1 = np.load(pathname+'/objvals_SVM_CG_method_1').reshape(5,1)
	objvals_method_2 = np.load(pathname+'/objvals_SVM_CG_method_2').reshape(5,1)
	objvals_method_3 = np.load(pathname+'/objvals_SVM_CG_method_3').reshape(5,1)


#---MANO 
	type_Sigma, P_list, k0, rho, tau_SNR = 2, [500, 1000, 2000, 5000, 10000], 10, 0.2, 1

	#times_SVM_CG_ADMM = np.array([[0.43], [1.82], [4.67], [15.07], [180]])



#---PLOT 1 Compare everybody
	legend_plot_1 = {0:'Compute path', 1:'Correl top 1k + T_max = 100', 2:'Correl top 50', 3:'No CG', 4:'ADMM'}
	times_list  = [times_method_1, times_method_2, times_method_3, times_method_0, times_SVM_CG_ADMM]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_large.pdf')

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_small.pdf')
	plt.close()


#---PLOT 2 Compare subgroup
	#legend_plot_2 = {0:'Correl top 1k + (tau='+str(0.5*scaling)+', T_max = 50)', 1:'Correl top 2k + (tau='+str(scaling)+', T_max = 50)', 2:'Correl top 50'}
	legend_plot_2 = {0:'Compute path', 1:'Correl top 1k + T_max = 100', 2:'Correl top 50'}
	times_list  = [times_method_1, times_method_2, times_method_3]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_large.pdf')

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_small.pdf')
	plt.close()



#---PLOT 1 Compare everybody
	legend_plot_3 = {0:'Compute path', 1:'Correl top 1k + T_max = 50', 2:'Correl top 50', 3:'No CG'}

	objvals_list  = [objvals_method_1, objvals_method_2, objvals_method_3, objvals_method_0]
	
	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_3, 'objval','large')
	plt.savefig(pathname+'/compare_objective_values_large.pdf')

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_3, 'objval','small')
	plt.savefig(pathname+'/compare_objective_values_small.pdf')
	plt.close()















#plot()



