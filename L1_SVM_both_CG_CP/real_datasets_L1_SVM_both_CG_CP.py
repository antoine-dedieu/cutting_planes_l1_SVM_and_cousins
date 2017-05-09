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

	dict_title = {1:'dexter', 2:'gisette', 3:'madelon'}

	name = str(DT.year)+'_'+str(DT.month)+'_'+str(DT.day)+'-'+str(DT.hour)+'h'+str(DT.minute)+'-'+str(dict_title[type_real_dataset])
	pathname=r'../../../L1_SVM_both_CG_CP_results/'+str(name)
	os.makedirs(pathname)


#---Load dataset
	f = open(pathname+'/results.txt', 'w')
	X_train, l2_X_train, y_train = real_dataset_read_data_uci(type_real_dataset, f)
	N, P = X_train.shape


	epsilon_RC  = 1e-2
	time_limit  = 300 

#---Define the path
	alpha_max    = np.max(np.sum( np.abs(X_train), axis=0))                 #infinite norm of the sum over the lines
	alpha_list   = [1e-2*alpha_max*0.7] #start with non empty support 



#---L1 SVM 
	write_and_print('\n###### alpha='+str(alpha)+' #####\n', f)
	write_and_print('\n###### L1 SVM without CG #####', f)
	beta_L1_SVM, support_L1_SVM, time_L1_SVM, index_CP = L1_SVM_CP(X_train, y_train, range(N), alpha, epsilon_RC, time_limit, f)


#---R L1 SVM 
	#write_and_print('\n###### L1 SVM with R: penalizedSVM #####', f)
	#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f)


#---R L1 SVM 
	#write_and_print('\n###### L1 SVM with R: SAM #####', f)
	#beta_R_L1_SVM, support_R_L1_SVM, time_R_L1_SVM = SAM_R_L1_SVM(X_train, y_train, alpha, f)



#---L1 SVM with CG
	n_samples  = np.max([100, N/16, P/16])    #run 4 times
	n_features = n_samples                    #run 4 times

	write_and_print('\n###### L1 SVM with CG #####', f)

#---1/ reduce number samples
	index_CP        = init_CP_dual(X_train, y_train, alpha, n_samples, f)
	X_train_reduced = np.array([X_train[i,:] for i in index_CP])
	y_train_reduced = np.array([y_train[i]   for i in index_CP])


#---2/ use CG and reduce now the columns
	index_SVM_CG = init_correlation(X_train_reduced, y_train_reduced, n_features, f)
	beta_SVM_CG, support_SVM_CG, time_SVM_CG_0, model_SVM_CG, index_SVM_CG = L1_SVM_CG(X_train_reduced, y_train_reduced, index_SVM_CG, alpha, epsilon_RC, time_limit, 0, f)

	X_train_reduced = np.array([X_train[:,j] for j in support_SVM_CG]).T


#---3/ use CP and reduce now the samples
	beta_SVM_CP, support_SVM_CP, time_SVM_CP, index_CP = L1_SVM_CP(X_train_reduced, y_train, index_CP, alpha, epsilon_RC, time_limit, f)

	X_train_reduced = np.array([X_train[i,:] for i in index_CP])
	y_train_reduced = np.array([y_train[i]   for i in index_CP])


#---4/ use CG and return the final estimator
	index_SVM_CG = init_correlation(X_train_reduced, y_train_reduced, n_features, f)
	beta_SVM_CG, support_SVM_CG, time_SVM_CG, model_SVM_CG, index_SVM_CG = L1_SVM_CG(X_train_reduced, y_train_reduced, index_SVM_CG, alpha, epsilon_RC, time_limit, 0, f)



#---5/ results
	same_support = len(set(support_SVM_CG)-set(support_L1_SVM))==0 and len(set(support_L1_SVM)-set(support_SVM_CG))==0
	compare_norm = np.linalg.norm(beta_SVM_CG - beta_L1_SVM)**2

	write_and_print('\nSame support:    '+str(same_support), f)
	write_and_print('Norm difference: '+str(compare_norm)+'\n\n', f)



real_datasets_L1_SVM_both_CG_CP(2)






