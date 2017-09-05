import numpy as np
import datetime
import os
import sys

from gurobipy import *

from L1_SVM_both_CG_CP import *
from L1_SVM_add_columns_delete_samples import *
from init_L1_SVM_both_CG_CP import *
 
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../algortihms')
from smoothing_hinge_loss import *

sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *
from R_L1_SVM import *
from L1_SVM_CG_plots import *
from compare_L1_SVM_CG import *
from benchmark import *


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
	loop_repeat = 2

	n_alpha_list = 1
	time_limit   = 60 

	n_features   = 50


#---Results
	times_L1_SVM               = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_SVM_CG_method_1      = [[ [] for i in range(n_alpha_list)] for P in N_P_list]  #delete the columns not in the support
	times_SVM_CG_method_2      = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_SVM_CG_method_3      = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_SVM_CG_method_4      = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	times_benchmark            = [[ [] for i in range(n_alpha_list)] for P in N_P_list]

	objvals_SVM_method_1       = [[ [] for i in range(n_alpha_list)] for P in N_P_list]
	objvals_SVM_method_2       = [[ [] for i in range(n_alpha_list)] for P in N_P_list]  
	objvals_SVM_method_3       = [[ [] for i in range(n_alpha_list)] for P in N_P_list]  
	objvals_SVM_method_4       = [[ [] for i in range(n_alpha_list)] for P in N_P_list] 
	objvals_benchmark          = [[ [] for i in range(n_alpha_list)] for P in N_P_list]  

	#same_supports    = [[] for N,P in N_P_list]
	#compare_norms    = [[] for N,P in N_P_list]


	aux=-1
	N_P_list_short    = []

	for N,P in N_P_list:

		write_and_print('\n\n\n-----------------------------For N='+str(N)+' and P='+str(P)+'--------------------------------------', f)

		aux += 1
		for i in range(loop_repeat):
			write_and_print('\n\n-------------------------SIMULATION '+str(i)+'--------------------------', f)

		#---Simulate data
			seed_X = random.randint(0,1000)

			if False:
				X_train, _, l2_X_train, y_train, _, _ = simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)
				np.save(r'../../../L1_SVM_both_CG_CP_results/X_train.npy',    X_train)
				np.save(r'../../../L1_SVM_both_CG_CP_results/l2_X_train.npy', l2_X_train)
				np.save(r'../../../L1_SVM_both_CG_CP_results/y_train.npy',    y_train)
			else:
				X_train    = np.load(r'../../../L1_SVM_both_CG_CP_results/X_train.npy')
				l2_X_train = np.load(r'../../../L1_SVM_both_CG_CP_results/l2_X_train.npy')
				y_train    = np.load(r'../../../L1_SVM_both_CG_CP_results/y_train.npy')
				print 'DATA LOADED'









			alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
			alpha_list   = [alpha_max*1e-3] #start with non empty support
			aux_alpha = -1


			n_features = 50
			#index_SVM_CG_correl, time_correl   = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
			#index_SVM_CG_liblinear, time_liblinear, beta_liblinear = liblinear_for_CG('squared_hinge_l1', X_train, y_train, alpha_list[0], f)



		#---METHOD 2: CORRELATION ON TOP 10N
			start_time = time.time()
			argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
			time_correl = time.time() - start_time

			N_columns = 10000
			index_CG        = argsort_columns[::-1][:N_columns]
			X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T
			
			tau_max = 0.2
			n_loop  = 1
			n_iter  = 200

			write_and_print('\n\n###### Method 2: tau='+str(tau_max)+' #####', f)
			#index_samples_method_2, index_columns_method_2_bis, time_smoothing_2, beta_method_2 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, n_iter, f)
			#index_columns_method_2 = np.array(index_CG)[index_columns_method_2_bis].tolist()
			

		#---METHOD 3
			#X_train_reduced_features = np.array([X_train[:,j] for j in index_columns_method_2]).T

			tau_max = 0.1
			n_loop  = 10
			n_iter  = 30

			#index_samples_method_3, index_columns_method_3_bis, time_smoothing_3_bis, beta_method_3_bis   = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced_features, y_train, alpha_list[0], tau_max, n_loop, n_iter, f)
			#index_columns_method_3 = np.array(index_columns_method_2)[index_columns_method_3_bis].tolist()



		#---NEW METHOD 4
			index_samples_method_4, index_columns_method_4, time_smoothing_4 = init_both_CG_CP_sampling_smoothing(X_train, y_train, alpha_list[0], f)




			model_L1_SVM      = 0
			model_method_1    = 0
			model_method_2    = 0
			model_method_3    = 0
			model_method_4    = 0


			for alpha in alpha_list:
				aux_alpha += 1
				write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

			#---L1 SVM 
				#write_and_print('\n###### L1 SVM with Gurobi without keeping model #####', f)
				#beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, 0, [], False, f) #_ = range(P) 


			#---L1 SVM with CG and not deleting
				#write_and_print('\n\n###### L1 SVM with CG correl 50 #####', f)
				#beta_method_1_supp, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1  = L1_SVM_CG(X_train, y_train, index_SVM_CG_correl, alpha, 1e-1, time_limit, model_method_1, [], False, f)	
				#times_SVM_CG_method_1[aux][aux_alpha].append(time_correl + time_method_1)
				#objvals_SVM_method_1[aux][aux_alpha].append(obj_val_method_1/float(obj_val_L1_SVM))


			#---Compute path
				#write_and_print('\n###### L1 SVM path with CG correl 10 #####', f)
				#alpha_bis = alpha_max
				#index_columns_method_1, _   = init_correlation(X_train, y_train, 10, f)
				#beta_method_1     = []
				#time_method_1_tot = 0

				#while 0.7*alpha_bis > alpha_list[0]:
				#	beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_bis, 1e-2, time_limit, model_method_1, beta_method_1, False, f)
				#	alpha_bis   *= 0.7
				#	time_method_1_tot += time_method_1
				#beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_list[0], 1e-2, time_limit, model_method_1, beta_method_1, False, f)
				#time_method_1_tot += time_method_1

				#times_SVM_CG_method_1[aux][aux_alpha].append(time_method_1_tot)
				#write_and_print('\nTIME ALL CG = '+str(time_method_1_tot), f)   


			#---L1 SVM with CG 
				#write_and_print('\n\n###### L1 SVM with CG-CP hinge AGD, method 2 + decrease tau 10*20  #####', f)
				write_and_print('\n\n###### L1 SVM with sampling approach #####', f)
				beta_method_4, support_method_4, time_method_4, model_method_4, index_samples_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_both_CG_CP(X_train, y_train, index_samples_method_4, index_columns_method_4, alpha, 1e-2, time_limit, model_method_4, [], False, f)
				times_SVM_CG_method_1[aux][aux_alpha].append((time_smoothing_4 + time_method_4))
				times_SVM_CG_method_3[aux][aux_alpha].append((time_smoothing_4 + time_method_4))


				if N < 100:
				#---L1 SVM with CG 
					#write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 1k, tau=1, T_max = 100 #####', f)
					write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 10N, tau=1, T_max = 100 #####', f)

					beta_method_2, support_method_2, time_method_2, model_method_2, index_columns_method_2, obj_val_method_2 = L1_SVM_CG(X_train, y_train, index_columns_method_2, alpha, 1e-1, time_limit, model_method_2, [], False, f)
					times_SVM_CG_method_2[aux][aux_alpha].append((time_correl + time_smoothing_2 + time_method_2))


				#---L1 SVM with CG 
					#write_and_print('\n\n###### L1 SVM with CG-CP hinge AGD, method 2 + decrease tau 10*20  #####', f)
					write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 10N, tau=1, T_max = 100 + decrease tau 10*20 #####', f)
					beta_method_3, support_method_3, time_method_3, model_method_3, index_samples_method_3, index_columns_method_3, obj_val_method_3 = L1_SVM_both_CG_CP(X_train, y_train, index_samples_method_3, index_columns_method_3, alpha, 1e-2, time_limit, model_method_3, [], False, f)
					times_SVM_CG_method_1[aux][aux_alpha].append((time_correl + time_smoothing_2 + time_smoothing_3_bis + time_method_3))
					times_SVM_CG_method_3[aux][aux_alpha].append((time_correl + time_smoothing_2 + time_smoothing_3_bis + time_method_3))




				#---BENCHMARK
					write_and_print('\n\n###### Benchmark AL_CD #####', f)
					#store_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, seed_X, 1e-2*alpha_max, 'double')
					#subprocess.call([pathname+'/../../best_subset_classification/LPsparse/LPsparse', '-d', pathname+'/../../best_subset_classification/LPsparse/data/synthetic_dataset/data_train_double'])
					#obj_val_AL_CD, time_AL_CD = check_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, 1e-2*alpha_max, os.path.dirname(os.path.realpath(__file__)), f)
					obj_val_AL_CD, time_AL_CD = 1,1
					times_benchmark[aux][aux_alpha].append(time_AL_CD)



				#---RELATIVE ACCURACY
					obj_val_method_1 = obj_val_method_2
					obj_val_L1_SVM = np.min([obj_val_method_1, obj_val_method_2, obj_val_method_3, obj_val_AL_CD]) + 1e-5

					objvals_SVM_method_1[aux][aux_alpha].append(obj_val_method_1/float(obj_val_L1_SVM) -1.)
					objvals_SVM_method_2[aux][aux_alpha].append(obj_val_method_2/float(obj_val_L1_SVM) -1.)
					objvals_SVM_method_3[aux][aux_alpha].append(obj_val_method_3/float(obj_val_L1_SVM) -1.)
					objvals_benchmark[aux][aux_alpha].append(obj_val_AL_CD/float(obj_val_L1_SVM) -1.)
				



#---Compare times
	#times_L1_SVM   = [ [ np.sum([times_L1_SVM[P][i][loop]             for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list_short))]
	#times_L1_SVM_WS= [ [ np.sum([times_L1_SVM_WS[P][i][loop]         for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(P_list))]
	times_method_1  = [ [ np.sum([times_SVM_CG_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	times_method_2  = [ [ np.sum([times_SVM_CG_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	times_method_3  = [ [ np.sum([times_SVM_CG_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	#times_method_4  = [ [ np.sum([times_SVM_CG_method_4[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	times_benchmark = [ [ np.sum([times_benchmark[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]


	np.save(pathname+'/times_method_1', times_method_1)
	np.save(pathname+'/times_method_2', times_method_2)
	np.save(pathname+'/times_method_3', times_SVM_CG_method_3)
	np.save(pathname+'/times_benchmark', times_benchmark)

#---PLOT 2 Compare everybody
	legend_plot_1 = {0:'RP CG', 1:'FO + CG', 2:'FO + C-CG', 3:'AL-CD'}
	times_list    = [times_method_1, times_method_2, times_method_3, times_benchmark]
	P_list        = [str(int(0.001*NP[1]))+'K' for NP in N_P_list]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar.pdf', bbox_inches='tight')
	plt.close()

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_1, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_small.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_small.pdf', bbox_inches='tight')
	plt.close()


#---PLOT 2 Subgroup 
	legend_plot_2 = {0:'RP CG', 1:'FO + CG', 2:'FO + C-CG'}
	times_list    = [times_method_1, times_method_2, times_method_3]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','large')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_subgroup_large.pdf', bbox_inches='tight')
	plt.close()

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, times_list, legend_plot_2, 'time','small')
	plt.savefig(pathname+'/compare_times_errorbar_subgroup_small.pdf', bbox_inches='tight')
	#plt.savefig('compare_times_errorbar_subgroup_small.pdf', bbox_inches='tight')
	plt.close()




#---Compare objective values
	#objvals_SVM_CG_method_0    = [ [1 for loop in range(loop_repeat)] for P in range(len(N_P_list_short))]
	objvals_SVM_CG_method_1    = [ [ np.mean([objvals_SVM_method_1[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	objvals_SVM_CG_method_2    = [ [ np.mean([objvals_SVM_method_2[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	objvals_SVM_CG_method_3    = [ [ np.mean([objvals_SVM_method_3[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]
	objvals_benchmark 		   = [ [ np.sum([objvals_benchmark[P][i][loop]    for i in range(n_alpha_list) ]) for loop in range(loop_repeat)] for P in range(len(N_P_list))]


	np.save(pathname+'/objvals_SVM_CG_method_1', objvals_SVM_CG_method_1)
	np.save(pathname+'/objvals_SVM_CG_method_2', objvals_SVM_CG_method_2)
	np.save(pathname+'/objvals_SVM_CG_method_3', objvals_SVM_CG_method_3)
	np.save(pathname+'/objvals_benchmark', objvals_benchmark)

	objvals_list    = [objvals_SVM_CG_method_1, objvals_SVM_CG_method_2, objvals_SVM_CG_method_3, objvals_benchmark]

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_1, 'objval','large')
	plt.savefig(pathname+'/compare_objvals_errorbar_large.pdf', bbox_inches='tight')
	#plt.savefig('compare_objvals_errorbar.pdf', bbox_inches='tight')
	plt.close()

	L1_SVM_plots_errorbar(type_Sigma, P_list, k0, rho, tau_SNR, objvals_list, legend_plot_1, 'objval','small')
	plt.savefig(pathname+'/compare_objvals_errorbar_small.pdf', bbox_inches='tight')
	plt.close()


#---PLOT 1 Compare everybody


#---PLOT 2 Compare subgroup
	#objvals_list  = [objvals_SVM_CG_method_1, objvals_SVM_CG_method_2, objvals_SVM_CG_method_3]
	#L1_SVM_plots_errorbar(type_Sigma, N_P_list, k0, rho, tau_SNR, objvals_list, legend_plot_2, 'objval')
	#plt.savefig(pathname+'/compare_objective_values_subgroup.pdf')
	#plt.close()











