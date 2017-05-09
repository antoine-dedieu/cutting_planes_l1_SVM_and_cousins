import numpy as np
import os

import sys
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


def process_Rahul_real_datasets():

	N_real, P_real = 58, 12625
	train_data   = open('datasets/radsens.x.txt',"r")
	train_labels = open('datasets/radsens.y.txt',"r")

	X_train = np.zeros((N_real, P_real))
	y_train = np.zeros(N_real)

	aux_line = -1
	for lines in train_data:
	    aux_line += 1
	    line = lines.split(" ")

	    aux_column = -1
	    for sign_float in line[1:]:
	        
	        if len(sign_float)>0:
	            aux_column += 1
	            X_train[aux_column][aux_line] = sign_float
	        else:
	            continue       

	aux = -1
	dict_classes = {1: -1, 2:1}
	for lines in train_labels:
	    line = lines.split(' ')
	    
	    for sign_float in line[1:]:
	        if len(sign_float)==1:
	            aux += 1
            	y_train[aux] = dict_classes[int(sign_float)]

	np.savetxt('datasets_processed/radsens/X_train.txt', X_train)
	np.savetxt('datasets_processed/radsens/y_train.txt', y_train)





def read_Rahul_real_dataset(f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    X_train = np.loadtxt(current_path+'/datasets_processed/radsens/X_train.txt')
    y_train = np.loadtxt(current_path+'/datasets_processed/radsens/y_train.txt')

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

    X_train = np.loadtxt(current_path+'/datasets_processed/radsens/X_train.txt')
    y_train = np.loadtxt(current_path+'/datasets_processed/radsens/y_train.txt')

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
