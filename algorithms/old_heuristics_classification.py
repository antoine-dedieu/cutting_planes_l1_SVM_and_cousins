import numpy as np
from sklearn.feature_selection import RFE
import time

import sys
sys.path.append('../algorithms')
from algorithm1_classification import *




def write_and_print(text,f):
    print text
    f.write('\n'+text)





def build_RFE_estimator(support, coefficients, P):
    #Builds the estimator on the whole support from RFE scikit result   
    beta_RFE_SVM = np.zeros(P)
    
    aux = -1
    for idx in np.where(support==True)[0]:
        aux+=1
        beta_RFE_SVM[idx] = coefficients[aux]
    return beta_RFE_SVM







def heuristic1(type_loss, type_penalization, X, y, K0_list, number_decreases, X_add, L, epsilon, time_limit, f):
    
#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2

    
    write_and_print('\n\nHEURISTIC 1 for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    
    N,P = X.shape[0],X.shape[1]
    start = time.time()

    
    #Results
    train_errors_K0_list = [[] for K0 in K0_list]
    betas_K0_list = [[] for K0 in K0_list]
       
    
#---INITIALIZATION

    dict_number_alpha_max={'hinge_l1': np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]), 
                           'squared_hinge_l1': 2*np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]),
                           'hinge_l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)]),
                           'squared_hinge_l2': 4*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    
    alpha_max = dict_number_alpha_max[str(type_loss)+'_'+str(type_penalization)]
    alpha_list = [alpha_max*1.2**(-i) for i in range(number_decreases)]


    
#---MAIN LOOP
    K0_min = K0_list[0]

    for K0 in K0_list:  
        
        #Initialize for k=0
        model_l1_l0 = 0
        model_status = 0
        beta_start = []
            
        for alpha in alpha_list:
            
            beta, beta0, train_error = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, L, epsilon, time_limit)   
    
            beta_start = beta0*np.ones(P+1)
            beta_start[:P] = beta

            train_errors_K0_list[K0-K0_min].append(train_error)
            betas_K0_list[K0-K0_min].append((beta,beta0)) 
        
        
    write_and_print('Time = '+str(round(time.time()-start,2)),f)
    return train_errors_K0_list, betas_K0_list, alpha_list









def heuristic2(type_descent, type_loss, type_penalization, X, y, K0_list, number_decreases, X_add, L, epsilon, time_limit, f):
    
#Type_descent: indicates which type of heuristic :
#  - if UP we increase k
#  - if DOWN we decrease k
    
#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2


    write_and_print('\n\nHEURISTIC 2 '+str(type_descent)+' for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    
    N,P = X.shape[0],X.shape[1]
    start = time.time()

    
    #Results
    train_errors_K0_list = [[] for K0 in K0_list]
    betas_K0_list = [[] for K0 in K0_list]
    l1_SVM_OR_RFE_list = []
       
    
#---INITIALIZATION

    dict_number_alpha_max={'hinge_l1': np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]), 
                           'squared_hinge_l1': 2*np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]),
                           'hinge_l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)]),
                           'squared_hinge_l2': 4*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    
    alpha_max = dict_number_alpha_max[str(type_loss)+'_'+str(type_penalization)]
    alpha_list = [alpha_max*1.2**(-i) for i in range(number_decreases)]

    
    
    
    
    
#---MAIN LOOP
    K0_min = K0_list[0]
    K0_max = K0_list[::-1][0]
    
    #K0 LIST
    if type_descent=='down':
        K0_list = K0_list[::-1]
    

    for alpha in alpha_list:   
        
        beta_start = []         

    #---IF GOING DOWN -> CHANGE WARM STARTS 
        if (type_descent=='down'):

            if type_loss=='hinge' and type_penalization=='l1' :
            
            #l1 SVM for same penalization -> we decide not to use previous result for initialization
                beta, beta0, _, _, _ = Gurobi_SVM('hinge', 'l1', 'no_l0', X, y, 0, alpha, time_limit, 0, 0)
                l1_SVM_OR_RFE_list.append((beta,beta0))


            else:
            
            #RFE k_max for l2 loss and same penalization, solved on dual -> we decide not to use previous result for initialization  
                dict_dual = {'l1':False, 'l2':True}
                
                estimator = svm.LinearSVC(penalty=type_penalization, loss= type_loss, dual=dict_dual[type_penalization], C=1/float(2*alpha))
                selector = RFE(estimator, n_features_to_select=K0_max, step=1)
                selector = selector.fit(X, y)

                support, coefficients = selector.support_, selector.estimator_.coef_[0]
                beta, beta0 = build_RFE_estimator(support, coefficients, P), selector.estimator_.intercept_[0]
                
                l1_SVM_OR_RFE_list.append((beta, beta0))
            
            beta_start = beta0*np.ones(P+1)
            beta_start[:P] =  beta      



    #---LOOP -> order has been changed
        for K0 in K0_list:  

            beta, beta0, train_error = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, L, epsilon, time_limit)   

            beta_start = beta0*np.ones(P+1)
            beta_start[:P] = beta       

            train_errors_K0_list[K0-K0_min].append(train_error)
            betas_K0_list[K0-K0_min].append((beta,beta0)) 

        
        
    write_and_print('Time = '+str(round(time.time()-start,2)),f)
    return train_errors_K0_list, betas_K0_list, alpha_list, l1_SVM_OR_RFE_list







def heuristic3(type_descent, type_loss, type_penalization, X, y, K0_list, number_decreases, X_add, L, epsilon, time_limit, f):
    
#Type_descent: indicates whihc type of heuristic :
#  - if UP we increase k
#  - if DOWN we decrease k


#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2

    
    write_and_print('\n\nHEURISTIC 3 '+str(type_descent)+' for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    
    N,P = X.shape[0],X.shape[1]
    start = time.time()

    
    #Results
    train_errors_K0_list = [[] for K0 in K0_list]
    betas_K0_list = [[] for K0 in K0_list]
    
    l1_SVM_OR_RFE_list = []
    l1_SVM_OR_RFE_errors_list = []
    
    #Understand the results
    accross_alpha, accross_k =[0 for K0 in K0_list],[0 for K0 in K0_list]
       
    
#---INITIALIZATION

    dict_number_alpha_max={'hinge_l1': np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]), 
                           'squared_hinge_l1': 2*np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]),
                           'hinge_l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)]),
                           'squared_hinge_l2': 4*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    
    alpha_max = dict_number_alpha_max[str(type_loss)+'_'+str(type_penalization)]
    alpha_list = [alpha_max*1.2**(-i) for i in range(number_decreases)]
    


        
#---WARM STARTS
    K0_min = K0_list[0]
    K0_max = K0_list[::-1][0]
    
    store_beta_starts_current_K0 = [[] for i in alpha_list]

    
    if type_descent == 'down': #CHANGE WARM STARTS
    
        loop = -1
        for alpha in alpha_list:
            
            if type_loss=='hinge' and type_penalization=='l1' : #-> we decide not to use previous result for initialization
                beta, beta0, _, _, _ = Gurobi_SVM('hinge', 'l1', 'no_l0', X, y, 0, alpha, time_limit, 0, 0)
                l1_SVM_OR_RFE_list.append((beta,beta0))


            else: #-> we decide not to use previous result for initialization
                dict_dual = {'l1':False, 'l2':True}
                
                estimator = svm.LinearSVC(penalty=type_penalization, loss= type_loss, dual=dict_dual[type_penalization], C=1/float(2*alpha))
                selector = RFE(estimator, n_features_to_select=K0_max, step=1)
                selector = selector.fit(X, y)

                support, coefficients = selector.support_, selector.estimator_.coef_[0]
                beta, beta0 = build_RFE_estimator(support, coefficients, P), selector.estimator_.intercept_[0]
                
                l1_SVM_OR_RFE_list.append((beta, beta0))
                
        
        
        
        #---store in both cases
            loop += 1
            beta_start = beta0*np.ones(P+1)
            beta_start[:P] =  beta      
            store_beta_starts_current_K0[loop] = beta_start
                
                


#---MAIN LOOP 
    
    #K0 LIST
    if type_descent == 'down':
        K0_list = K0_list[::-1]

    for K0 in K0_list: #list has been reversed if type_descent = 1

        beta_start = [] 

        #Two lists needed to store beta starts
        store_beta_starts_previous_K0 = store_beta_starts_current_K0
        store_beta_starts_current_K0 = []


        loop = -1

        for alpha in alpha_list:
            print '\nK0='+str(K0)+'  alpha='+str(round(alpha,2))
            loop += 1

            beta_1, beta0_1, train_error_1 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start, X_add, L, epsilon, time_limit)   
            beta_2, beta0_2, train_error_2 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, store_beta_starts_previous_K0[loop], X_add, L, epsilon, time_limit)   
                

        #---Case 1: move across k
            if(train_error_1<train_error_2):
                accross_k[K0-K0_min]+=1
                betas_K0_list[K0-K0_min].append((beta_1, beta0_1))
                train_errors_K0_list[K0-K0_min].append(train_error_1)

                beta_start = beta0_1*np.ones(P+1)
                beta_start[:P] = beta_1
                store_beta_starts_current_K0.append(beta_start)


        #---Case 2: move across alpha  
            else:
                accross_alpha[K0-K0_min]+=1
                betas_K0_list[K0-K0_min].append((beta_2, beta0_2))
                train_errors_K0_list[K0-K0_min].append(train_error_2)

                beta_start = beta0_2*np.ones(P+1)
                beta_start[:P] = beta_2
                store_beta_starts_current_K0.append(beta_start)



    write_and_print('Moves across alpha : '+str(accross_alpha),f)
    write_and_print('Moves across K     : '+str(accross_k),f)
    write_and_print('Time = '+str(round(time.time()-start,2)),f)
    

    return train_errors_K0_list, betas_K0_list, alpha_list, l1_SVM_OR_RFE_list









def all_RFE_estimators(type_loss, type_penalization, X, y, K0_list, number_decreases, X_add, L, epsilon, time_limit, f):
    
#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2

    
    write_and_print('\n\nRFE 1 for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    
    N,P = X.shape[0],X.shape[1]
    start = time.time()

    
    #Results
    train_errors_K0_list = [[] for K0 in K0_list]
    betas_K0_list = [[] for K0 in K0_list]
       
    
#---INITIALIZATION

    dict_number_alpha_max={'hinge_l1': N*np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]), 
                           'squared_hinge_l1': 2*np.max([np.sum([abs(X[i][j]) for i in range(N)]) for j in range(P)]),
                           'hinge_l2': 2*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)]),
                           'squared_hinge_l2': 4*N*np.max([np.linalg.norm(X[i,:])**2 for i in range(N)])}
    
    alpha_max = dict_number_alpha_max[str(type_loss)+'_'+str(type_penalization)]
    alpha_list = [alpha_max*1.2**(-i) for i in range(number_decreases)]


    
#---MAIN LOOP
    dict_dual = {'l1':False, 'l2':True}
    K0_min = K0_list[0]

    for alpha in alpha_list:
        estimator = svm.LinearSVC(penalty=type_penalization, loss= type_loss, dual=dict_dual[type_penalization], C=1/float(2*alpha))
        selector = RFE(estimator, n_features_to_select=1, step=1)
        selector = selector.fit(X, y)


    #---TRAIN AGAIN ON SUPPORT -> not optimal
        list_ranking = selector.ranking_.tolist()
        support = np.zeros(P)

        for K0 in K0_list:

            if(K0>0):
                idx = list_ranking.index(K0)
                support[idx] = 1
                beta, beta0, train_error = SVM_support(type_loss, type_penalization, X, y, alpha, support, time_limit)

                train_errors_K0_list[K0-K0_min].append(train_error)
                betas_K0_list[K0-K0_min].append((beta,beta0)) 

            else:
                train_errors_K0_list[0].append(np.sum(y))
                betas_K0_list[0].append(  (np.zeros(P),0)   )
        
    write_and_print('Time = '+str(round(time.time()-start,2)),f)
    
    return train_errors_K0_list, betas_K0_list






def best_of_up_down(train_errors_K0_list_H3, train_errors_K0_list_H3b, betas_K0_list_H3, betas_K0_list_H3b, K0_list, number_decreases, f):
    
    write_and_print('\nBEST OF UP / DOWN ?', f)

    #Results
    best_betas = [[] for K0 in K0_list]
    best_train_errors = [[] for K0 in K0_list]
    list_3, list_3b = [],[]

    for K0 in K0_list:
        
        c,d=0,0

        for loop in range(number_decreases):
            a = train_errors_K0_list_H3[K0][loop]
            b = train_errors_K0_list_H3b[K0][loop]

            if(a<b):
                c+=1
                best_train_errors[K0].append(a)
                best_betas[K0].append(betas_K0_list_H3[K0][loop])
            else:
                d+=1
                best_train_errors[K0].append(b)
                best_betas[K0].append(betas_K0_list_H3b[K0][loop])

        list_3.append(c)
        list_3b.append(d)
     
    write_and_print('UP    : '+str(list_3), f)
    write_and_print('DOwN  : '+str(list_3b), f)

    return best_train_errors, best_betas


