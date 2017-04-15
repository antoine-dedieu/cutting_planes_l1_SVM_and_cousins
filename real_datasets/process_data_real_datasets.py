import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle

import os



def write_and_print(text,f):
    print text
    f.write('\n'+text)





def process_data_real_datasets(type_real_dataset):
    
    if(type_real_dataset==1):
        g=open('datasets/lungCancer_train.data',"r")
        h=open('datasets/lungCancer_test.data',"r")
        dict_type = {'Mesothelioma\r\n':-1,'ADCA\r\n':1}

    elif(type_real_dataset==2):
        g=open('datasets/leukemia_train.data',"r")
        h=open('datasets/leukemia_test.data',"r")
        dict_type = {'ALL\r\r\n':-1,'AML\r\r\n':1}


#-------------------X--------------------
    X0, y = pd.DataFrame(), []

    for i in [g,h]:
        for line in i:
            line,data_line=line.split(",")[::-1],[]
            y.append(dict_type[str(line[0])])

            for aux in line[1:len(line)]:
                data_line.append(float(aux))
            X0=pd.concat([X0,pd.DataFrame(data_line).T])

    N,P = X0.shape
    X0.index = range(N)
    X0.columns = range(P)
    X = X0.values
    #X0 = shuffle(X0)

    for i in range(P):
        X[:,i] = X[:,i]- np.mean(X[:,i])


    N = N/2
    X_train, X_test = X[:N], X[N:]
    y_train, y_test = y[:N], y[N:]


    print 'Train size : '+str(X_train.shape)
    print 'Test size : '+str(X_test.shape)

    size = 0
    dict_title = {1:'lungCancer', 2:'leukemia'}
    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/X_train_'+str(size)+'.txt', X_train)
    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/X_test_'+str(size)+'.txt',  X_test)

    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train_'+str(size)+'.txt', y_train)
    np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/y_test_'+str(size)+'.txt',  y_test)







def real_dataset_read_data(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'lungCancer', 2:'leukemia'}
    size = 0

    X_train = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/X_train_'+str(size)+'.txt')
    X_test  = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/X_test_'+str(size)+'.txt')

    y_train = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train_'+str(size)+'.txt')
    y_test  = np.loadtxt(current_path+'/datasets_processed/'+str(dict_title[type_real_dataset])+'/y_test_'+str(size)+'.txt')


    write_and_print('Train size : '+str(X_train.shape),f)
    write_and_print('Test size : '+str(X_test.shape),f)



#------------SNR, EPSILON and Y------------- 

    N_train,P = X_train.shape
    N_test, P = X_test.shape
    

#------------NORMALIZE------------- 
    
#---Normalize all the X columns
    #l2_y_train = np.linalg.norm(y_train)
    #y_train = y_train/float(l2_y_train)

    l2_X_train   = []

    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2)


    print 'DATA CREATED'
    f.write('DATA CREATED')

    return X_train, X_test, l2_X_train, y_train, y_test



#process_data_real_datasets(1)
#process_data_real_datasets(2)




