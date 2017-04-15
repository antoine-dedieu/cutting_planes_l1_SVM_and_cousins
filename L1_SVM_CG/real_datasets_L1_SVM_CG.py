import datetime
import os
import sys

from L1_SVM_CG import *
from R_L1_SVM import *


sys.path.append('../real_datasets')
from process_data_real_datasets import *



def real_datasets_L1_SVM_CG(type_real_dataset):

# TYPE_REAL_DATASET = 1 : TRIAZINE DATASET
# TYPE_REAL_DATASET = 2 : RIBOFLAVIN DATASET

	
	DT = datetime.datetime.now()

	dict_title ={1:'leukemia',2:'lung_cancer'}
	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_CG_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	X_train, X_test, l2_X_train, y_train, y_test = real_dataset_read_data(type_real_dataset, f)
	N, P = X_train.shape


	epsilon_RC  = 1e-2
	alpha       = 1
	time_limit  = 30  



#---L1 SVM 
	write_and_print('\n###### L1 SVM with Gurobi without CG #####', f)
	beta_L1_SVM, support_L1_SVM, time_L1_SVM, _, _  = L1_SVM_CG(X_train, y_train, range(P), alpha, epsilon_RC, time_limit, 0, f)


#---R L1 SVM 
	write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
	beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)



#---R L1 SVM 
	#write_and_print('\n###### L1 SVM with R: SAM #####', f)
	#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)


#---L1 SVM with CG
	n_features = N
	write_and_print('\n###### L1 SVM with CG #####', f)
	#index_CG = RFE_for_CG(X_train, y_train, alpha, n_features, f)
	index_SVM_CG = init_correlation(X_train, y_train, n_features, f)
	beta_SVM_CG, support_SVM_CG, time_SVM_CG, _, _ = L1_SVM_CG(X_train, y_train, index_SVM_CG, alpha, epsilon_RC, time_limit, 0, f)


	same_support = len(set(support_SVM_CG)-set(support_R_L1_SVM))==0 and len(set(support_R_L1_SVM)-set(support_SVM_CG))==0
	compare_norm = np.linalg.norm(beta_SVM_CG - beta_R_L1_SVM)**2



	write_and_print('\nSame support:    '+str(same_support), f)
	write_and_print('Norm difference: '+str(compare_norm)+'\n\n', f)



real_datasets_L1_SVM_CG(1)
real_datasets_L1_SVM_CG(2)






