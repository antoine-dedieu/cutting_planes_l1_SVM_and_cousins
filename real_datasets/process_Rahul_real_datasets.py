import numpy as np
import os

import sys
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


def process_Rahul_real_datasets(type_real_dataset):

	if type_real_dataset ==3:
		N_real, P_real = 58, 12625
		train_data   = open('../../../datasets/datasets_unprocessed/radsens.x.txt',"r")
		train_labels = open('../../../datasets/datasets_unprocessed/radsens.y.txt',"r")
	
	if type_real_dataset ==6:
		N_real, P_real = 198, 16063
		train_data = open('../../../datasets/datasets_unprocessed/14cancer.xtrain.txt',"r")
        valid_data = open('../../../datasets/datasets_unprocessed/14cancer.xtest.txt',"r")

        train_labels = open('../../../datasets/datasets_unprocessed/14cancer.ytrain.txt',"r")
        valid_labels = open('../../../datasets/datasets_unprocessed/14cancer.ytest.txt',"r")


	X = np.zeros((N_real, P_real))
	y = np.zeros(N_real)

#---All
	aux_line = -1
	for lines in train_data:
		aux_line += 1
		line = lines.split(" ")

		aux_column = -1
		for sign_float in line[1:]:
			
			if len(sign_float)>0:
				aux_column += 1
				X[aux_column][aux_line] = sign_float
			else:
				continue

#---14 cancer
	if type_real_dataset ==6:
		for lines in valid_data:
			aux_line += 1
			line = lines.split(" ")

			aux_column = -1
			for sign_float in line[1:]:
				
				if len(sign_float)>0:
					aux_column += 1
					X[aux_column][aux_line] = sign_float
				else:
					continue    

#---All
	aux = -1
	dict_classes = {1: -1, 2:1}
	for lines in train_labels:
		line = lines.split(' ')
		
		for sign_float in line[1:]:
			if len(sign_float)==1:
				aux += 1
				y_train[aux] = dict_classes[int(sign_float)]



	for i in range(P_real):
        #print np.mean(X[:,i])
		X[:,i] -= np.mean(X[:,i])
		X[:,i] /= np.linalg.norm(X[:,i] + 1e-10)
		
	if type_real_dataset ==3:
		np.savetxt('../../../datasets/datasets_processed/radsens/X.txt', X_train)
		np.savetxt('../../../datasets/datasets_processed/radsens/y.txt', y_train)
	if type_real_dataset ==6:
		np.savetxt('../../../datasets/datasets_processed/14cancer/X.txt', X_train)
		np.savetxt('../../../datasets/datasets_processed/14cancer/y.txt', y_train)



def split_Rahul_real_dataset(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {3:'radsens', 4:'14cancer'}
    size = 0

    X   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt')
    y   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt')
    N,P = X.shape

    return X, y, 0





def split_Rahul_real_dataset_bis(type_real_dataset, f):

	current_path = os.path.dirname(os.path.realpath(__file__))

	dict_title = {3:'radsens', 4:'14cancer'}

	X   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt')
	y   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt')
	N,P = X.shape



#---RANDOMLY SPLIT
	y = np.array(y)
	idx_plus  = np.where(y==  1)[0]
	idx_minus = np.where(y== -1)[0]

	len_plus  = len(idx_plus)
	len_minus = len(idx_minus)


	seed = random.randint(0,1000)
	random.seed(a=seed)
	idx_plus_train  = random.sample(xrange(len_plus), len_plus/2)
	idx_minus_train = random.sample(xrange(len_minus),len_minus/2)

	idx_plus_test  = list(set(range(len_plus))  - set(idx_plus_train))
	idx_minus_test = list(set(range(len_minus)) - set(idx_minus_train))

	X_train = np.concatenate([ X[idx_plus[idx_plus_train],:], X[idx_minus[idx_minus_train],:] ])
	y_train = np.concatenate([ y[idx_plus[idx_plus_train]],   y[idx_minus[idx_minus_train]] ])

	X_test = np.concatenate([ X[idx_plus[idx_plus_test],:], X[idx_minus[idx_minus_test],:] ])
	y_test = np.concatenate([ y[idx_plus[idx_plus_test]],   y[idx_minus[idx_minus_test]] ])


	write_and_print('Train size : '+str(X_train.shape),f)
	write_and_print('Test size : '+str(X_test.shape),f)


#------------NORMALIZE------------- 
    

	l2_X_train   = []
	for i in range(P):
		l2 = np.linalg.norm(X_train[:,i])
		l2_X_train.append(l2)        
		X_train[:,i] = X_train[:,i]/float(l2)


	return X_train, X_test, y_train, y_test, seed, l2_X_train 










def read_Rahul_real_dataset(f):

	dict_title = {3:'radsens', 2:'leukemia'}

	current_path = os.path.dirname(os.path.realpath(__file__))

	X_train = np.loadtxt(current_path+'/../../datasets_processed/radsens/X_train.txt')
	y_train = np.loadtxt(current_path+'/../../datasets_processed/radsens/y_train.txt')

	N_train, P = X_train.shape
	write_and_print('Train size : '+str((N_train, P)),f)

   

#------------NORMALIZE------------- 

	l2_X_train   = []

	for i in range(P):
		l2 = np.linalg.norm(X_train[:,i])
		l2_X_train.append(l2)        
		X_train[:,i] = X_train[:,i]/float(l2+1e-5)


	print 'DATA CREATED'
	f.write('DATA CREATED')

	return X_train, l2_X_train, y_train







def read_Rahul_real_dataset_SVM_light_format():

	current_path = os.path.dirname(os.path.realpath(__file__))

	X_train = np.loadtxt(current_path+'/../../datasets_processed/radsens/X_train.txt')
	y_train = np.loadtxt(current_path+'/../../datasets_processed/radsens/y_train.txt')

	N_train, P = X_train.shape
	print 'Train size : '+str((N_train, P))

   

#------------NORMALIZE------------- 

	l2_X_train   = []

	for i in range(P):
		l2 = np.linalg.norm(X_train[:,i])
		l2_X_train.append(l2)        
		X_train[:,i] = X_train[:,i]/float(l2+1e-5)


#---------STORE INTO GOOD FORMAT------------- 

	data_train = open(current_path+'/../../struct_svm_admm/data/radsens/data_train', 'w')
	data_test  = open(current_path+'/../../struct_svm_admm/data/radsens/data_test', 'w')

	for i in range(N_train):
		line  = str(int(1.5+0.5*y_train[i]))+' '
		for j in range(P):
			line += str(j+1)+':'+str(X_train[i,j])+' '
		line += '\n'
		data_train.write(line)



# read_Rahul_real_dataset_SVM_light_format()
#process_Rahul_real_datasets()
