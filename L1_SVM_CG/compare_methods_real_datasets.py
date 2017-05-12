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


#---Store for elastic net
	store_ADMM_SVM_comparison(X_train, y_train, 'real', 'train')
	store_ADMM_SVM_comparison(X_test,  y_test,  'real', 'test')


	N, P = X_train.shape
	epsilon_RC  = 1e-2



#---REGULARIZATION
	alpha_max_L1    = np.max(np.sum( np.abs(X_train), axis=0))  #infinite norm of the sum over thealpha        = 1e-2*alpha_max
	alpha_max_L2    = np.max([ np.sum([X_train[i,j]**2 for j in range(P)]) for i in range(N) ])

	n_alpha_list = 60
	alpha_list_L1   = [alpha_max_L1*0.7**i for i in np.arange(1, n_alpha_list+1)]
	alpha_list_L2   = [alpha_max_L2*0.7**i for i in np.arange(1, n_alpha_list+1)]


#---RESULTS
	times_L1_SVM = []
	times_L2_SVM = []
	times_EN_SVM = []

	misclassification_L1 = []
	misclassification_L2 = []
	misclassification_EN = []

	size_support_L1 = []
	size_support_L2 = []
	size_support_EN = []


#---MAIN LOOP
	model_method_L1   = 0	
	n_features        = 10
	index_method_L1,_ = init_correlation(X_train, y_train, n_features, f)

	aux_alpha = -1

	for idx in range(n_alpha_list):

		aux_alpha += 1
		alpha_L1   = alpha_list_L1[idx]
		alpha_L2   = alpha_list_L2[idx]

		write_and_print('\n------------Alpha='+str(alpha)+'-------------', f)

	#---L1 SVM 
		beta_method_L1    = np.zeros(P) if aux_alpha == 0 else []
		write_and_print('\n###### L1 SVM with CG correlation, eps=1e-2 #####', f)
		beta_method_L1, support_method_L1, time_method_L1, model_method_L1, index_method_L1, obj_val_method_L1  = L1_SVM_CG(X_train, y_train, index_method_L1_SVM, alpha, 1e-2, time_limit, model_method_L1, beta_method_L1, False, f)
		
		classifications =  np.sign(np.dot(X_test[:, support_method_L1], beta_method_L1[0]) + beta_method_L1[1]*np.ones(N_test))
		mis_score       = np.sum(-0.5*y_test*classifications+0.5*np.ones(N_test))

		size_support_L1.append(len(support_method_3))
		misclassification_L1.append(mis_score)
		times_L1_SVM.append(time_method_3)


	#---L2 SVM 
		write_and_print('\n###### L2 SVM with liblinear #####', f)
		beta_liblinear, support_liblinear, time_liblinear = liblinear_for_CG('hinge_l2', X_train, y_train, alpha, f)

		classifications =  np.sign(np.dot(X_test[:, support_liblinear], beta_liblinear[0]) + beta_liblinear[1]*np.ones(N_test))
		mis_score       = np.sum(-0.5*y_test*classifications+0.5*np.ones(N_test))

		size_support_L2.append(len(support_liblinear))
		misclassification_L2.append(mis_score)
		times_L2_SVM.append(time_liblinear)



	#---EN SVM
		subprocess.call([pathname+'/../../best_subset_classification/struct_svm_admm/structsvm_sdm_learn', '-c', str(1./alpha), '-w', '3', pathname+'/../../best_subset_classification/LPsparse/data/real_dataset/data_train', '_', pathname+'/../../best_subset_classification/LPsparse/data/real_dataset/data_test'])
		accuracy_EN, time_EN = store_ADMM_SVM_comparison()
		misclassification_EN.append(1. - accuracy_EN/100.)
		times_EN_SVM.append(time_EN)



	best_idx_L1 = np.argmin(misclassification_L1)
	best_idx_L2 = np.argmin(misclassification_L2)
	best_idx_EN = np.argmin(misclassification_EN)


	write_and_print('\nTime L1 SVM                  :'+str(np.sum(times_L1_SVM)), f)
	write_and_print('Best misclassifcation L1 SVM :'+str(misclassification_L1[best_idx_L1]), f)
	write_and_print('Size best support     L1 SVM :'+str(size_support_L1[best_idx_L1]), f)

	write_and_print('\nTime L2 SVM                  :'+str(np.sum(times_L2_SVM)), f)
	write_and_print('Best misclassifcation L2 SVM :'+str(misclassification_L2[best_idx_L2]), f)
	write_and_print('Size best support     L2 SVM :'+str(size_support_L2[best_idx_L2]), f)

	write_and_print('\nTime EN SVM                  :'+str(np.sum(times_EN_SVM)), f)
	write_and_print('Best misclassifcation EN SVM :'+str(misclassification_EN[best_idx_EN]), f)
	#write_and_print('Size best support     EN SVM :'+str(size_support_[best_idx_EN]), f)



compare_methods_real_datasets(1)
compare_methods_real_datasets(2)	
#compare_methods_real_datasets(3)  






