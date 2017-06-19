import numpy as np
import random
from sklearn.utils import shuffle

import os



def write_and_print(text,f):
    print text
    f.write('\n'+text)





def process_data_real_datasets(type_real_dataset):

    if(type_real_dataset==0):
        g=open('../../../datasets/datasets_unprocessed/ovarian.data',"r")
        dict_type = {'Normal\n':-1,'Cancer\n':1}
    
    if(type_real_dataset==1):
        g=open('../../../datasets/datasets_unprocessed/lungCancer_train.data',"r")
        h=open('../../../datasets/datasets_unprocessed/lungCancer_test.data',"r")
        dict_type = {'Mesothelioma\r\n':-1,'ADCA\r\n':1}

    elif(type_real_dataset==2):
        g=open('../../../datasets/datasets_unprocessed/leukemia_train.data',"r")
        h=open('../../../datasets/datasets_unprocessed/leukemia_test.data',"r")
        dict_type = {'ALL\r\r\n':-1,'AML\r\r\n':1}


#-------------------X--------------------
    X0, y = pd.DataFrame(), []



    if type_real_dataset>0:
        for i in [g,h]:
            for line in i:
                line,data_line=line.split(",")[::-1],[]
                y.append(dict_type[str(line[0])])

                for aux in line[1:len(line)]:
                    data_line.append(float(aux))
                X0=pd.concat([X0,pd.DataFrame(data_line).T])

    else:
        for line in g:
            line,data_line=line.split(",")[::-1],[]
            y.append(dict_type[str(line[0])])
            
            for aux in line[1:len(line)]:
                data_line.append(float(aux))
            X0=pd.concat([X0,pd.DataFrame(data_line).T])


    N,P = X0.shape
    X0.index = range(N)
    X0.columns = range(P)
    X = X0.values
    
    print N, P

    for i in range(P):
        X[:,i] -= np.mean(X[:,i])
        #X[:,i] /= np.linalg.norm(X[:,i])


    dict_title = {0:'ovarian', 1:'lungCancer', 2:'leukemia'}
    np.savetxt('../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt', X)
    np.savetxt('../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt', y)

    #np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train_'+str(size)+'.txt', y_train)
    #np.savetxt('datasets_processed/'+str(dict_title[type_real_dataset])+'/y_test_'+str(size)+'.txt',  y_test)



def split_real_dataset(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {0:'ovarian', 1:'lungCancer', 2:'leukemia'}
    size = 0

    X   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/X.txt')
    y   = np.loadtxt(current_path+'/../../../datasets/datasets_processed/'+str(dict_title[type_real_dataset])+'/y.txt')
    N,P = X.shape

    return X, y, 0



def split_real_dataset_bis(type_real_dataset, f):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'lungCancer', 2:'leukemia'}
    size = 0

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










def real_dataset_SVM_light_format(type_real_dataset):

    current_path = os.path.dirname(os.path.realpath(__file__))

    dict_title = {1:'lungCancer', 2:'leukemia'}
    size = 0

    X_train = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/X_train_'+str(size)+'.txt')
    X_test  = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/X_test_'+str(size)+'.txt')

    y_train = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/y_train_'+str(size)+'.txt')
    y_test  = np.loadtxt(current_path+'/../../datasets_processed/'+str(dict_title[type_real_dataset])+'/y_test_'+str(size)+'.txt')


    print 'Train size : '+str(X_train.shape)
    print 'Test size : '+str(X_test.shape)



#------------SNR, EPSILON and Y------------- 

    N_train,P = X_train.shape
    N_test, P = X_test.shape
    

#------------NORMALIZE------------- 
    
#---Normalize all the X columns
    #l2_y_train = np.linalg.norm(y_train)
    #y_train = y_train/float(l2_y_train)

    l2_X_train   = []

    #for i in range(P):
    #    l2 = np.linalg.norm(X_train[:,i])
    #    l2_X_train.append(l2)        
    #    X_train[:,i] = X_train[:,i]/float(l2)




#---------STORE INTO GOOD FORMAT------------- 

    data_train = open(current_path+'/../../struct_svm_admm/data/'+str(dict_title[type_real_dataset])+'/data_train', 'w')
    data_test  = open(current_path+'/../../struct_svm_admm/data/'+str(dict_title[type_real_dataset])+'/data_test', 'w')

    for i in range(N_train):
        line  = str(int(1.5+0.5*y_train[i]))+' '
        for j in range(P):
            line += str(j+1)+':'+str(X_train[i,j])+' '
        line += '\n'
        data_train.write(line)


    for i in range(N_test):
        line  = str(int(1.5+0.5*y_test[i]))+' '
        for j in range(P):
            line += str(j+1)+':'+str(X_test[i,j])+' '
        line += '\n'
        data_test.write(line)


#process_data_real_datasets(0)
#process_data_real_datasets(1)
#process_data_real_datasets(2)



#real_dataset_SVM_light_format(1)
#real_dataset_SVM_light_format(2)




