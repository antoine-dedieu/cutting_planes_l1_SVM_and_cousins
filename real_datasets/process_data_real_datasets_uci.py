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
        train_data = open('../../datasets/dexter_train.data.txt',"r")
        valid_data = open('../../datasets/dexter_valid.data.txt',"r")

        train_labels   = open('../../datasets/dexter_train.labels.txt',"r")
        valid_labels = open('../../datasets/dexter_valid.labels.txt',"r")

    elif type_real_dataset == 2:
        N_real, P_real = 7000, 5000
        train_data = open('../../datasets/gisette_train.data.txt',"r")
        valid_data = open('../../datasets/gisette_valid.data.txt',"r")

        train_labels = open('../../datasets/gisette_train.labels.txt',"r")
        valid_labels = open('../../datasets/gisette_valid.labels.txt',"r")
    
    
    elif type_real_dataset == 3:
        N_real, P_real = 7000, 5000
        train_data = open('../../datasets/madelon_train.data.txt',"r")
        valid_data = open('../../datasets/madelon_valid.data.txt',"r")

        train_labels = open('../../datasets/madelon_train.labels.txt',"r")
        valid_labels = open('../../datasets/madelon_valid.labels.txt',"r")
    
    elif type_real_dataset == 4:
        N_real, P_real = 200, 100000
        train_data = open('../../datasets/arcene_train.data.txt',"r")
        valid_data = open('../../datasets/arcene_valid.data.txt',"r")

        train_labels = open('../../datasets/arcene_train.labels.txt',"r")
        valid_labels = open('../../datasets/arcene_valid.labels.txt',"r")

    X = np.zeros((N_real, P_real))
    y = np.zeros(N_real)

    aux = -1
    for lines in train_data:
        aux += 1
        line = lines.split(" ")
        
        for j in range(len(line)):
            if str(line[j])!='\n': #if line not ended
                X[aux, j] = int(line[j])
            else:
                break
                

    for lines in valid_data:
        aux += 1
        line = lines.split(" ")

        for j in range(len(line)):
            if str(line[j])!='\n': #if line not ended
                X[aux, j] = int(line[j])
            else:
                break

    aux = -1
    for lines in train_labels:
        aux += 1
        y[aux] = lines
    for lines in valid_labels:
        aux += 1
        y[aux] = lines

    for i in range(P_real):
        X[:,i] -= np.mean(X[:,i])
        
    dict_title = {1:'dexter', 2:'gisette', 3:'madelon', 4:'arcene'}
    np.savetxt('../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt', X)
    np.savetxt('../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt', y)




def split_real_dataset_uci(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'dexter', 2:'gisette', 3:'madelon', 4:'arcene'}
    size = 0

    y   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt')
    X   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt')
    N,P = X.shape

    return X, y, 0



def split_real_dataset_uci_bis(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'dexter', 2:'gisette', 3:'madelon', 4:'arcene'}
    size = 0

    y   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt')
    X   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt')
    N,P = X.shape


    #y_train = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train_'+str(size)+'.txt')
    #y_test  = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/y_test_'+str(size)+'.txt')



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



#------------SNR, EPSILON and Y------------- 

    N_train,P = X_train.shape
    N_test, P = X_test.shape
    

#------------NORMALIZE------------- 

    l2_X_train   = []
    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2)



    return X_train, X_test, y_train, y_test, seed, l2_X_train



#process_data_real_datasets_uci(1)
#process_data_real_datasets_uci(2)
#process_data_real_datasets_uci(4)







