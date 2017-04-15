import datetime
import os
import sys

sys.path.append('../real_datasets')
from process_data_real_datasets_uci import *

sys.path.append('../L1_SVM_CG')
from L1_SVM_CG import *
from R_L1_SVM import *

sys.path.append('../L1_SVM_CP')
from L1_SVM_CP import *
from init_L1_SVM_CP import *



def real_datasets_L1_SVM_both_CG_CP_path(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET

	
	DT = datetime.datetime.now()

	dict_title = {1:'dexter', 2:'gisette', 3:'madelon'}

	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_both_CG_CP_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	X_train, l2_X_train, y_train = real_dataset_read_data_uci(type_real_dataset, f)
	N, P = X_train.shape


#---Paramters
	n_alpha_list = 20
	epsilon_RC   = 1e-2
	n_features   = 10
	time_limit   = 300


#---Define the path
	alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
	alpha_list   = [alpha_max*0.7**i for i in np.arange(1, n_alpha_list+1)] #start with non empty support
	delete_samples = True




#---Results
	times_L1_SVM               = []
	times_SVM_CG_no_delete     = []
	times_SVM_add_delete       = []  #delete the columns not in the support

	objvals_SVM_CG_no_delete    = []
	objvals_SVM_add_delete      = []



#---Warm start Gurobi
	index_SVM_CG_no_delete = init_correlation(X_train, y_train, n_features, f) #just for highest alpha
	index_columns          = init_correlation(X_train, y_train, n_features, f)
	index_samples          = range(N) #all constraints for lambda max


	model_L1_SVM           = 0
	model_SVM_CG_no_delete = 0
	model_SVM_add_delete   = 0



	aux_alpha = -1
	for alpha in alpha_list:
		aux_alpha += 1
		write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)


	#---L1 SVM 
		write_and_print('\n\n###### L1 SVM#####', f)
		beta_L1_SVM, support_L1_SVM, time_L1_SVM, model_L1_SVM, _, objval_L1_SVM = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, model_L1_SVM, False, f)
		times_L1_SVM.append(time_L1_SVM/N)


	#---L1 SVM with CG and not deleting
		write_and_print('\n\n###### L1 SVM with CG and no deletion #####', f)
		beta_SVM, support_SVM, time_SVM_CG_no_delete, model_SVM_CG_no_delete, index_SVM_CG_no_delete, objval_SVM_CG_no_delete = L1_SVM_CG(X_train, y_train, index_SVM_CG_no_delete, alpha, epsilon_RC, time_limit, model_SVM_CG_no_delete, False, f)
		times_SVM_CG_no_delete.append(time_SVM_CG_no_delete/N)
		objvals_SVM_CG_no_delete.append(objval_SVM_CG_no_delete/objval_L1_SVM)
		print(N, len(index_SVM_CG_no_delete))


	#---L1 SVM with CG and not deleting
		write_and_print('\n\n###### L1 SVM with column generation and constraint deletion #####', f)
		beta_SVM, support_SVM, time_SVM_add_delete, model_SVM_add_delete, index_samples, index_columns, delete_samples, objval_add_delete = L1_SVM_add_columns_delete_samples(X_train, y_train, index_samples, index_columns, alpha, epsilon_RC, time_limit, model_SVM_add_delete, delete_samples, f)
		times_SVM_add_delete.append(time_SVM_add_delete/N)
		objvals_SVM_add_delete.append(objval_add_delete/objval_L1_SVM)
		print len(index_samples), len(index_columns) 


	write_and_print('\n\n###### Time L1 SVM: '+str(np.sum(times_L1_SVM))+' #####', f)
	write_and_print('###### Time CG    : '+str(np.sum(times_L1_SVM))+' #####', f)
	write_and_print('###### Time CG CD : '+str(np.sum(times_L1_SVM))+' #####', f)

	write_and_print('\n\n###### Ratio CG    : '+str(np.mean(objvals_SVM_CG_no_delete))+' #####', f)
	write_and_print('###### Ratio CG CD : '+str(np.mean(objvals_SVM_add_delete))+' #####', f)


real_datasets_L1_SVM_both_CG_CP_path(1)






