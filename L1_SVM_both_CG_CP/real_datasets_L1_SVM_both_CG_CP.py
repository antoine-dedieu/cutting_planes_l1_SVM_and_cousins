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



def real_datasets_L1_SVM_both_CG_CP(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET

	
	DT = datetime.datetime.now()

	dict_title = {-1:'farm-ads', 1:'dexter', 2:'gisette', 3:'madelon'}

	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_both_CG_CP_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	#X_train, l2_X_train, y_train = real_dataset_read_data_uci(type_real_dataset, f)
	X_train, y_train = real_dataset_read_data(type_real_dataset)
	N, P = X_train.shape


	epsilon_RC  = 1e-2
	time_limit  = 300 

#---Define the path
	alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
	alpha_list   = [1e-2*alpha_max*0.7] #start with non empty support 



#---Column and constraint Generation
	index_samples_method_4, index_columns_method_4, time_smoothing_4 = init_both_CG_CP_sampling_smoothing(X_train, y_train, alpha_list[0], f)
	beta_method_4, support_method_4, time_method_4, model_method_4, index_samples_method_4, index_columns_method_4, obj_val_method_4 = L1_SVM_both_CG_CP(X_train, y_train, index_samples_method_4, index_columns_method_4, alpha, 1e-2, time_limit, model_method_4, [], False, f)
				
	write_and_print('\nTime C-CG'+str(time_method_4), f)



real_datasets_L1_SVM_both_CG_CP(-1)






