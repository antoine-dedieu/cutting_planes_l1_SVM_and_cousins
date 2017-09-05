


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