import datetime
import os
import sys

from L1_SVM_CG import *
from R_L1_SVM import *

sys.path.append('../algortihms')
from smoothing_hinge_loss import *


sys.path.append('../real_datasets')
from process_data_real_datasets import *
from process_Rahul_real_datasets import *



def real_datasets_L1_SVM_CG_path(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET
# TYPE_REAL_DATASET = 2 : RADSENS

	
	DT = datetime.datetime.now()

	dict_title ={1:'lung_cancer',2:'leukemia',3:'radsens'}
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

	n_alpha_list = 15
	alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)]
	#alpha_list   = [1e-2*alpha_max]

	time_limit  = 30  




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

	n_features = 10



#---MAIN LOOP
	model_L1_SVM      = 0
	model_method_1    = 0
	model_method_2    = 0
	model_method_3    = 0
	model_method_4    = 0


	
	index_method_1,_    = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
	index_method_2,_    = init_correlation(X_train, y_train, n_features, f)
	index_method_3,_    = init_correlation(X_train, y_train, n_features, f)



	aux_alpha = -1

	for alpha in alpha_list:

		aux_alpha += 1
		write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

	#---L1 SVM 
		write_and_print('\n###### L1 SVM with Gurobi without CG without warm start #####', f)
		beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, obj_val_L1_SVM  = L1_SVM_CG(X_train, y_train, range(P), alpha, 0, time_limit, model_L1_SVM, [], False, f)
		times_L1_SVM.append(time_L1_SVM)



	#---L1 SVM with CG 
		beta_method_1    = np.zeros(P) if aux_alpha == 0 else []
		write_and_print('\n###### L1 SVM with CG correlation, eps=1 #####', f)
		beta_method_1, support_method_1, time_method_1, model_method_1, index_method_1, obj_val_method_1   = L1_SVM_CG(X_train, y_train, index_method_1, alpha, 5e-1, time_limit, model_method_1, beta_method_1, False, f)
		times_SVM_CG_method_1.append(time_method_1)
		objvals_SVM_method_1.append(obj_val_method_1/float(obj_val_L1_SVM))


	#---L1 SVM with CG 
		beta_method_2    = np.zeros(P) if aux_alpha == 0 else []
		write_and_print('\n###### L1 SVM with CG correlation, eps=1e-1 #####', f)
		beta_method_2, support_method_2, time_method_2, model_method_2, index_method_2, obj_val_method_2  = L1_SVM_CG(X_train, y_train, index_method_2, alpha, 1e-1, time_limit, model_method_2, beta_method_2, False, f)
		times_SVM_CG_method_2.append(time_method_2)
		objvals_SVM_method_2.append(obj_val_method_2/float(obj_val_L1_SVM))

	#---L1 SVM with CG 
		beta_method_3    = np.zeros(P) if aux_alpha == 0 else []
		write_and_print('\n###### L1 SVM with CG correlation, eps=1e-2 #####', f)
		beta_method_3, support_method_3, time_method_3, model_method_3, index_method_3, obj_val_method_3  = L1_SVM_CG(X_train, y_train, index_method_3, alpha, 1e-2, time_limit, model_method_3, beta_method_3, False, f)
		times_SVM_CG_method_3.append(time_method_3)
		objvals_SVM_method_3.append(obj_val_method_3/float(obj_val_L1_SVM))
			


	write_and_print('Time Gurobi without CG             :'+str(np.sum(times_L1_SVM)), f)
	write_and_print('Time CG with correlation, eps=5e-1 :'+str(np.sum(times_SVM_CG_method_1)), f)
	write_and_print('Time CG with correlation, eps=1e-1 :'+str(np.sum(times_SVM_CG_method_2)), f)
	write_and_print('Time CG with correlation, eps=1e-2 :'+str(np.sum(times_SVM_CG_method_3)), f)

	write_and_print('Ratio CG with correlation, eps=5e-1 :'+str(np.mean(objvals_SVM_method_1)), f)
	write_and_print('Ratio CG with correlation, eps=1e-1 :'+str(np.mean(objvals_SVM_method_2)), f)
	write_and_print('Ratio CG with correlation, eps=1e-2 :'+str(np.mean(objvals_SVM_method_3)), f)



real_datasets_L1_SVM_CG_path(1)
real_datasets_L1_SVM_CG_path(2)	
real_datasets_L1_SVM_CG_path(3)  






