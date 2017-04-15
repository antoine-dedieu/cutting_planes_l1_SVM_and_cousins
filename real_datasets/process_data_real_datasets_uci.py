import numpy as np
import random
from sklearn.utils import shuffle

import os



def write_and_print(text,f):
    print text
    f.write('\n'+text)


def process_data_real_datasets_uci(type_real_dataset):

    if type_real_dataset == 1:
        N_real, P_real = 600, 20000
        train_data = open('datasets/dexter_train.data.txt',"r")
        valid_data = open('datasets/dexter_valid.data.txt',"r")

        train_labels   = open('datasets/dexter_train.labels.txt',"r")
        valid_labels = open('datasets/dexter_valid.labels.txt',"r")

    elif type_real_dataset == 2:
        N_real, P_real = 7000, 5000
        train_data = open('datasets/gisette_train.data.txt',"r")
        valid_data = open('datasets/gisette_valid.data.txt',"r")

        train_labels = open('datasets/gisette_train.labels.txt',"r")
        valid_labels = open('datasets/gisette_valid.labels.txt',"r")
    
    
    elif type_real_dataset == 3:
        N_real, P_real = 7000, 5000
        train_data = open('datasets/madelon_train.data.txt',"r")
        valid_data = open('datasets/madelon_valid.data.txt',"r")

        train_labels = open('datasets/madelon_train.labels.txt',"r")
        valid_labels = open('datasets/madelon_valid.labels.txt',"r")
    


    X_train = np.zeros((N_real, P_real))
    y_train = np.zeros(N_real)

    aux = -1
    for lines in train_data:
        aux += 1
        line = lines.split(" ")
        
        for couples in line:

            if str(couples)!='\n': #if line not ended
                if type_real_dataset == 1:
                    couple = couples.split(":")
                    X_train[aux, int(couple[0])] = couple[1]
                if type_real_dataset == 2:
                    X_train[int(couples)] = 1
            else:
                break
                

    for lines in valid_data:
        aux += 1
        line = lines.split(" ")

        for couples in line:

            if str(couples)!='\n': #if line not ended
                if type_real_dataset == 1:
                    couple = couples.split(":")
                    X_train[aux, int(couple[0])] = couple[1]
                if type_real_dataset == 2:
                    X_train[int(couples)] = 1
            else:
                break
                

    aux = -1
    for lines in train_labels:
        aux += 1
        y_train[aux] = lines
    for lines in valid_labels:
        aux += 1
        y_train[aux] = lines
        
    dict_title = {1:'dexter', 2:'gisette', 3:'madelon'}
    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/X_train.txt', X_train)
    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train.txt', y_train)








def real_dataset_read_data_uci(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'dexter', 2:'gisette', 3:'madelon'}

    X_train = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/X_train.txt')
    y_train = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train.txt')
    write_and_print('Train size : '+str(X_train.shape),f)


#------------SNR, EPSILON and Y------------- 

    N_train,P = X_train.shape
    

#------------NORMALIZE------------- 

    l2_X_train   = []

    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2+1e-5)


    print 'DATA CREATED'
    f.write('DATA CREATED')

    return X_train, l2_X_train, y_train



#process_data_real_datasets_uci(1)
#process_data_real_datasets_uci(2)








