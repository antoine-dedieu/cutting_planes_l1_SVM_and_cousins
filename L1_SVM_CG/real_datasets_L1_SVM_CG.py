import datetime
import os
import sys

from L1_SVM_CG import *
from R_L1_SVM import *
from compare_L1_SVM_CG import *

sys.path.append('../algortihms')
from smoothing_hinge_loss import *


sys.path.append('../real_datasets')
from process_data_real_datasets import *
from process_Rahul_real_datasets import *



def real_datasets_L1_SVM_CG(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET
# TYPE_REAL_DATASET = 2 : RADSENS

	
	DT = datetime.datetime.now()

	dict_title ={1:'leukemia',2:'lung_cancer',3:'radsens'}
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_CG_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	if type_real_dataset<3:
		X_train, X_test, l2_X_train, y_train, y_test = real_dataset_read_data(type_real_dataset, f)
	elif type_real_dataset==3:
		X_train, l2_X_train, y_train = read_Rahul_real_dataset(f)
	

	N, P = X_train.shape

	epsilon_RC  = 1e-2

	alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))  #infinite norm of the sum over thealpha        = 1e-2*alpha_max

	n_alpha_list = 10
	#alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)]
	alpha_list   = [1e-2*alpha_max]

	time_limit  = 30  

	store_ADMM_SVM_comparison(X_train, y_train, N, P, seed_X, 1e-2*alpha_max)




#---RESULTS
	times_L1_SVM               = []
	times_L1_SVM_WS            = []
	times_SVM_CG_method_1      = []  #delete the columns not in the support
	times_SVM_CG_method_2      = []
	times_SVM_CG_method_3      = []
	times_SVM_CG_method_4      = []

	objvals_SVM_method_1       = []
	objvals_SVM_method_2       = [] 
	objvals_SVM_method_3       = [] 
	objvals_SVM_method_4       = [] 

	n_features = 50



#---MAIN LOOP
	model_L1_SVM      = 0
	model_method_1    = 0
	model_method_2    = 0
	model_method_3    = 0
	model_method_4    = 0


	
	

#---METHOD 1
	index_SVM_CG_correl, time_correl   = init_correlation(X_train, y_train, n_features, f)


#---METHOD 2
	scaling = round ( (200./N)**2, 2)

	argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
	index_CG        = argsort_columns[::-1][:1000]
	X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T


	tau_max = 0.5*scaling
	n_loop  = 1
	n_iter  = 200

	write_and_print('\n\n###### Method 2 #####', f)
	index_samples_method_2, index_columns_method_2, time_smoothing_2, beta_method_2 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
	index_columns_method_2 = np.array(index_CG)[index_columns_method_2].tolist()
	


#---METHOD 3
	argsort_columns = np.argsort(np.abs(np.dot(X_train.T, y_train) ))
	index_CG        = argsort_columns[::-1][:2000]
	X_train_reduced = np.array([X_train[:,j] for j in index_CG]).T

	tau_max = 1*scaling
	n_loop  = 1
	n_iter  = 200

	write_and_print('\n\n###### Method 3 #####', f)

	index_samples_method_3, index_columns_method_3, time_smoothing_3, beta_method_3 = loop_smoothing_hinge_loss('hinge', 'l1', X_train_reduced, y_train, alpha_list[0], tau_max, n_loop, time_limit, n_iter, f)
	index_columns_method_3 = np.array(index_CG)[index_columns_method_3].tolist()


	aux_alpha = -1

	for alpha in alpha_list:

		write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

		aux_alpha += 1

	#---L1 SVM 
		write_and_print('\n###### L1 SVM with Gurobi without CG without warm start #####', f)
		beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, obj_val_L1_SVM  = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM, [], False, f)
		times_L1_SVM.append(time_L1_SVM)


	#---L1 SVM WS
		#beta_L1_SVM_WS    = np.zeros(P) if aux_alpha == 0 else []
		#write_and_print('\n###### L1 SVM with Gurobi without CG with warm start #####', f)
		#beta_L1_SVM_WS, support_L1_SVM, time_L1_SVM, model_L1_SVM_WS, _, obj_val_L1_SVM = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM_WS, beta_L1_SVM_WS, False, f) #_ = range(P) 
		#times_L1_SVM_WS.append(time_L1_SVM)


	#---R L1 SVM 
		#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
		#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)



	#---R L1 SVM 
		#write_and_print('\n###### L1 SVM with R: SAM #####', f)
		#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)


	#---Compute path
		write_and_print('\n###### L1 SVM path with CG correl 10 #####', f)
		alpha_bis = alpha_max
		index_columns_method_1, _   = init_correlation(X_train, y_train, 10, f)
		beta_method_1     = []
		time_method_1_tot = 0

		while 0.7*alpha_bis > alpha_list[0]:
			beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_bis, 1e-2, time_limit, model_method_1, beta_method_1, False, f)
			alpha_bis   *= 0.7
			time_method_1_tot += time_method_1
		beta_method_1, support_method_1, time_method_1, model_method_1, index_columns_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_columns_method_1, alpha_list[0], 1e-2, time_limit, model_method_1, beta_method_1, False, f)
		time_method_1_tot += time_method_1

		times_SVM_CG_method_1.append(time_method_1_tot)
		objvals_SVM_method_1.append(obj_val_method_1/float(obj_val_L1_SVM))

		write_and_print('\nTIME ALL CG = '+str(time_method_1_tot), f)


	#---METHOD 1
		write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 1k, tau='+str(0.5*scaling)+', T_max = 50 #####', f)
		beta_method_2, support_method_2, time_method_2, model_method_2, index_method_2, obj_val_method_2 = L1_SVM_CG(X_train, y_train, index_columns_method_2, alpha, 1e-2, time_limit, model_method_2, [], False, f)
		times_SVM_CG_method_2.append(time_correl + time_smoothing_2 + time_method_2)
		objvals_SVM_method_2.append(obj_val_method_2/ float(obj_val_L1_SVM))


	#---METHOD 2
		write_and_print('\n\n###### L1 SVM with CG hinge AGD, correl 2k, tau='+str(scaling)+', T_max = 50 #####', f)
		beta_method_3, support_method_3, time_method_3, model_method_3, index_columns_method_3, obj_val_method_3 = L1_SVM_CG(X_train, y_train, index_columns_method_3, alpha, 1e-2, time_limit, model_method_3, [], False, f)
		times_SVM_CG_method_3.append(time_correl + time_smoothing_3 + time_method_3)
		objvals_SVM_method_3.append(obj_val_method_3/ float(obj_val_L1_SVM))


	#---L1 SVM with CG and liblinear
		#beta_method_2    = np.zeros(P) if aux_alpha == 0 else []
		#write_and_print('\n###### L1 SVM with CG and liblinear #####', f)
		#beta_method_2, support_method_2, time_method_2, model_method_2, index_method_2, obj_val_method_2 = L1_SVM_CG(X_train, y_train, index_method_2, alpha, 1e-2, time_limit, model_method_2, beta_method_2, False, f)
		#times_SVM_CG_method_2.append(time_method_2)
		#objvals_SVM_method_2.append(obj_val_method_2/ float(obj_val_L1_SVM))


	#---L1 SVM with CG and not deleting
		#write_and_print('\n\n###### L1 SVM with CG hinge AGD, eps=5e-2, T_max = 200  #####', f)
		#write_and_print(str(sorted(index_columns_method_3)), f)
		#beta_method_3, support_method_3, time_method_3, model_method_3, index_columns_method_3, obj_val_method_3 = L1_SVM_CG(X_train, y_train, index_columns_method_3, alpha, 5e-2, time_limit, model_method_3, beta_method_3, False, f)
		#times_SVM_CG_method_3.append(time_init_3 + time_method_3)
		#objvals_SVM_method_3.append(obj_val_method_3/float(obj_val_L1_SVM))


	#---L1 SVM with CG and not deleting
		write_and_print('\n\n###### L1 SVM with CG correl 50 #####', f)
		beta_method_4, support_method_4, time_method_4, model_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_CG(X_train, y_train, index_SVM_CG_correl, alpha, 5e-2, time_limit, model_method_4, [], False, f)
		times_SVM_CG_method_4.append(time_correl + time_method_4)
		objvals_SVM_method_4.append(obj_val_method_4/float(obj_val_L1_SVM))
					





	write_and_print('Time Gurobi without CG without warm start              :'+str(np.sum(times_L1_SVM)), f)
	write_and_print('Time CG with correl path                       :'+str(np.sum(times_SVM_CG_method_1)), f)
	write_and_print('Time CG hinge AGD, correl 1k, tau='+str(0.5*scaling)+', T_max = 50 :'+str(np.sum(times_SVM_CG_method_2)), f)
	write_and_print('Time CG hinge AGD, correl 2k, tau='+str(scaling)+', T_max = 50   :'+str(np.sum(times_SVM_CG_method_3)), f)
	write_and_print('Time CG correl 50                                    :'+str(np.sum(times_SVM_CG_method_4)), f)
	#write_and_print('\nMean ratio objval CG with correlation, eps=1e-2         :'+str(np.sum(objvals_SVM_method_2)), f)
	
	write_and_print('Ratio CG with correl path                       :'+str(np.sum(objvals_SVM_method_1)), f)
	write_and_print('Ratio CG hinge AGD, correl 1k, tau='+str(0.5*scaling)+', T_max = 50 :'+str(np.sum(objvals_SVM_method_2)), f)
	write_and_print('Ratio CG hinge AGD, correl 2k, tau='+str(scaling)+', T_max = 50   :'+str(np.sum(objvals_SVM_method_3)), f)
	write_and_print('Ratio CG correl 50                                    :'+str(np.sum(objvals_SVM_method_4)), f)


#real_datasets_L1_SVM_CG(1)
real_datasets_L1_SVM_CG(2)	#BAD -> too small and all features are in top 50
#real_datasets_L1_SVM_CG(3)  #BAD -> correlation add 50






