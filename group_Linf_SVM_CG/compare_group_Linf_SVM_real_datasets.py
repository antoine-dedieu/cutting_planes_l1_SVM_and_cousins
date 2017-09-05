import datetime
import os
import sys
import subprocess

from L1_SVM_CG import *
from R_L1_SVM import *
from benchmark import *

sys.path.append('../algortihms')
from smoothing_hinge_loss import *


sys.path.append('../real_datasets')
from process_data_real_datasets import *
from process_Rahul_real_datasets import *
from process_data_real_datasets_uci import *



def compare_L1_SVM_real_datasets(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET
# TYPE_REAL_DATASET = 2 : RADSENS

	
	DT = datetime.datetime.now()

	dict_title ={0:'ovarian', 1:'lung_cancer',2:'leukemia',3:'radsens',4:'arcene'}
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_CG_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	loop_repeat = 5


#---Before loop
	times_L1_SVM               = []
	times_SVM_CG_method_1      = []
	times_SVM_CG_method_3      = []
	times_SVM_CG_method_4      = []
	times_benchmark            = []


	for i in range(loop_repeat):
#---REPETITIONS
	
		if type_real_dataset==0:
			X_train, y_train, seed_X = split_real_dataset(type_real_dataset, f)
		elif type_real_dataset==1 or type_real_dataset==2:
			X_train, X_test, y_train, y_test, seed_X = split_real_dataset_bis(type_real_dataset, f)
			N_test = X_test.shape[0]
			print y_train, y_test
			#X_train, y_train, seed_X = split_real_dataset(type_real_dataset, f)
		elif type_real_dataset==3:
			X_train, y_train, seed_X = split_Rahul_real_dataset(type_real_dataset, f)
		elif type_real_dataset==4:
			#X_train, y_train, seed_X = split_real_dataset_uci(type_real_dataset, f)
			X_train, X_test, y_train, y_test, seed_X = split_real_dataset_uci_bis(type_real_dataset, f)




		write_and_print('\n\n\n------------DATASET '+str(dict_title[type_real_dataset])+'-------------', f)
		write_and_print('\n\n\n------------SEED '+str(seed_X)+'-------------', f)


		N, P        = X_train.shape
		epsilon_RC  = 1e-2
		time_limit  = 20

		l2_X_train = []
		for i in range(P):
			X_train[:,i] -= np.mean(X_train[:,i])
			l2_X          = np.linalg.norm(X_train[:,i])
			X_train[:,i] /= l2_X +1e-10
			l2_X_train.append(l2_X)


	#---Store for elastic net
		#store_ADMM_SVM_comparison(X_train, y_train, 'real', 'train')
		#store_ADMM_SVM_comparison(X_test, y_test, 'real', 'test')



	#---REGULARIZATION
		alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))  #infinite norm of the sum over thealpha        = 1e-2*alpha_max

		n_alpha_list = 1
		alpha_list   = [1e-2*alpha_max]



	#---METHOD 1
		n_features = 50
		index_SVM_CG_correl, time_correl   = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
		index_SVM_CG_liblinear, time_liblinear, beta_liblinear = liblinear_for_CG('squared_hinge_l1', X_train, y_train, alpha_list[0], f)
		


	#---METHOD 3
		write_and_print('\n\n###### Method 3 #####', f)
		argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
		index_CG        = argsort_columns[::-1][:10*N]
		X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T

		tau_max = 0.2
		n_loop  = 1
		n_iter  = 400

		index_samples_method_3, index_columns_method_3, time_smoothing_3, beta_method_3 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
		index_columns_method_3 = np.array(index_CG)[index_columns_method_3].tolist()



	#---MODELS
		model_L1_SVM      = 0
		model_method_1    = 0
		model_method_2    = 0
		model_method_3    = 0
		model_method_4    = 0

		aux_alpha = -1

		for idx in range(n_alpha_list):

			aux_alpha += 1
			alpha   = alpha_list[idx]

			write_and_print('\n------------Alpha L1 '+str(alpha)+'-------------', f)

			#---L1 SVM 
			write_and_print('\n###### L1 SVM with Gurobi without CG without warm start #####', f)
			if P<1e5:
				beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha_list[0], 0, time_limit, model_L1_SVM, [], False, f) #_ = range(P) 
			else: 
				time_L1_SVM = 1
			times_L1_SVM.append(time_L1_SVM)


		#---Compute path
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

			times_SVM_CG_method_1.append(time_method_1_tot)

			write_and_print('\nTIME ALL CG = '+str(time_method_1_tot), f)



		#---L1 SVM with CG and not deleting
			write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 10n, tau='+str(1)+', T_max = 100 #####', f)
			beta_method_3, support_method_3, time_method_3, model_method_3, index_columns_method_3, obj_val_method_3 = L1_SVM_CG(X_train, y_train, index_columns_method_3, alpha_list[0], 1e-2, time_limit, model_method_3, [], False, f)
			times_SVM_CG_method_3.append(time_correl + time_smoothing_3 + time_method_3)


		#---L1 SVM with correlation
			write_and_print('\n\n###### L1 SVM with CG correl 50 #####', f)
			beta_method_4, support_method_4, time_method_4, model_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_CG(X_train, y_train, index_SVM_CG_correl, alpha_list[0], 1e-2, time_limit, model_method_4, [], False, f)
			times_SVM_CG_method_4.append(time_correl + time_method_4)



		#---BENCHMARK
			write_and_print('\n\n###### Benchmark AL_CD #####', f)
			if type_real_dataset==4:
				l2_X_train = np.ones(P)
				alpha_max /= 5
			store_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, seed_X, 5e-3*alpha_max, 'double')
			subprocess.call([pathname+'/../../best_subset_classification/LPsparse/LPsparse', '-d', pathname+'/../../best_subset_classification/LPsparse/data/synthetic_dataset/data_train_double'])
			obj_val_AL_CD, time_AL_CD = check_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, 5e-3*alpha_max, os.path.dirname(os.path.realpath(__file__)), f)

			times_benchmark.append(time_AL_CD)



	write_and_print('Time Gurobi          :'+str(np.mean(times_L1_SVM)), f)
	write_and_print('Time CG Reg path     :'+str(np.mean(times_SVM_CG_method_1)), f)
	write_and_print('Time CG AGD SHL + CG :'+str(np.mean(times_SVM_CG_method_3)), f)
	write_and_print('Time CG correl 50    :'+str(np.mean(times_SVM_CG_method_4)), f)
	write_and_print('Time CG AL-CD        :'+str(np.mean(times_benchmark)), f)

	write_and_print('Std Gurobi          :'+str(np.std(times_L1_SVM)), f)
	write_and_print('Std CG Reg path     :'+str(np.std(times_SVM_CG_method_1)), f)
	write_and_print('Std CG AGD SHL + CG :'+str(np.std(times_SVM_CG_method_3)), f)
	write_and_print('Std CG correl 50    :'+str(np.std(times_SVM_CG_method_4)), f)
	write_and_print('Std CG AL-CD        :'+str(np.std(times_benchmark)), f)


#compare_L1_SVM_real_datasets(0)  
#compare_L1_SVM_real_datasets(1)
#compare_L1_SVM_real_datasets(2)	
compare_L1_SVM_real_datasets(3)  
compare_L1_SVM_real_datasets(4)  #correl better





