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



def compare_methods_real_datasets(type_real_dataset):

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
		N_test = X_test.shape[0]
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
	times_L2_SVM               = []
	times_SVM_CG_method_1      = []  #delete the columns not in the support
	times_SVM_CG_method_2      = []
	times_SVM_CG_method_3      = []
	times_SVM_CG_method_4      = []

	



#---MAIN LOOP
	model_method_3   = 0	
	n_features       = 10
	index_method_3,_ = init_correlation(X_train, y_train, n_features, f)



	misclassification_L1 = []
	misclassification_L2 = []

	size_support_L1 = []
	size_support_L2 = []



	aux_alpha = -1

	for alpha in alpha_list:

		aux_alpha += 1
		write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)


	#---L2 SVM 
		write_and_print('\n###### L2 SVM with liblinear #####', f)
		beta_liblinear, support_liblinear, time_liblinear = liblinear_for_CG('hinge_l2', X_train, y_train, alpha, f)

		classifications =  np.sign(np.dot(X_test[:, support_liblinear], beta_liblinear[0]) + beta_liblinear[1]*np.ones(N_test))
		mis_score       = np.sum(-0.5*y_test*classifications+0.5*np.ones(N_test))

		size_support_L2.append(len(support_liblinear))
		misclassification_L2.append(mis_score)
		times_L2_SVM.append(time_liblinear)



	#---L1 SVM 
		beta_method_3    = np.zeros(P) if aux_alpha == 0 else []
		write_and_print('\n###### L1 SVM with CG correlation, eps=1e-2 #####', f)
		beta_method_3, support_method_3, time_method_3, model_method_3, index_method_3, obj_val_method_3  = L1_SVM_CG(X_train, y_train, index_method_3, alpha, 1e-2, time_limit, model_method_3, beta_method_3, False, f)
		
		classifications =  np.sign(np.dot(X_test[:, support_method_3], beta_method_3[0]) + beta_method_3[1]*np.ones(N_test))
		mis_score       = np.sum(-0.5*y_test*classifications+0.5*np.ones(N_test))

		size_support_L1.append(len(support_method_3))
		misclassification_L1.append(mis_score)
		times_SVM_CG_method_3.append(time_method_3)
			


	best_idx_L1 = np.argmin(misclassification_L1)
	best_idx_L2 = np.argmin(misclassification_L2)


	write_and_print('\nTime L2 SVM                  :'+str(np.sum(times_L2_SVM)), f)
	write_and_print('Best misclassifcation L2 SVM :'+str(misclassification_L2[best_idx_L2]), f)
	write_and_print('Size best support     L2 SVM :'+str(size_support_L2[best_idx_L2]), f)


	write_and_print('\nTime L1 SVM                  :'+str(np.sum(times_SVM_CG_method_3)), f)
	write_and_print('Best misclassifcation L1 SVM :'+str(misclassification_L1[best_idx_L1]), f)
	write_and_print('Size best support     L1 SVM :'+str(size_support_L1[best_idx_L1]), f)



compare_methods_real_datasets(1)
compare_methods_real_datasets(2)	
#compare_methods_real_datasets(3)  






