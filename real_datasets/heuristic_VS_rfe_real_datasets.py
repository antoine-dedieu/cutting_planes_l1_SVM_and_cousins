import numpy as np

import sys
sys.path.append('../algorithms')
from heuristics_classification import *




def heuristic_VS_rfe_real_datasets(type_loss, X_train, X_test, y_train, y_test, betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1, K0_list, f):

#K0_SUBLIST : USED TO PLOT THE RESULTS -> for every penalization, K0 for which we have the minima
#TEST_ERROR_LIST, GOOD_VARIABLES_LIST : USED TO SEE IF WE BEAT LASSO

#For every penalization, we want to :
#1/ Have the best K0 to plot the 3 metrics and see if we beat Lasso
#2/ For every K0, ave all the test errors and VS errors on different supports to plot and see if we beat Lasso
#3/ For every K0, see if the LS decreases, if the best support in test set is different to the last one


    N_train,P = X_train.shape
    N_test,P = X_test.shape
    
    K0_min = K0_list[0]
    
    
    dict_loss = {'squared_hinge':'SQUARED HINGE',
                'hinge':'HINGE'}
    
    list_title = ['HEURISTIC L2','HEURISTIC L1', 'RFE L2', 'RFE L1']
    
    
    aux = -1

    for betas in [betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1]:
        aux += 1
        write_and_print('\n\n------------------------------------------------------\nFOR '+str(dict_loss[type_loss])+' AND '+str(list_title[aux]) ,f)
    
    #---Best results
        best_beta = []
        best_beta_support = []
        best_K0 = -1
        best_test_error = 1e6
       
    
    
    #---COMPUTE
                       
        for K0 in K0_list:
            write_and_print('\n\nK0 = '+str(K0), f)
           
            train_errors = [1e10] 
            test_errors = [1e10] 

            betas_non_null = [np.zeros(P)] 
                       

            for loop in range(len(betas[K0])):
 
                beta, beta0 = betas[K0][loop]
                support = set(np.where(beta!=0)[0]) 

            #---Errors
                if len(support) == K0:
                    dot_product_y_train = y_train*(np.dot(X_train,beta)+ beta0*np.ones(N_train))
                    dot_product_y_test = y_test*(np.dot(X_test,beta)+ beta0*np.ones(N_test))

                    dict_loss_train = {'hinge': np.sum([max(0, 1-dot_product_y_train[i])  for i in range(N_train)]),
                                       'squared_hinge': np.sum([max(0, 1-dot_product_y_train[i])**2  for i in range(N_train)]) }

                    dict_loss_test = {'hinge': np.sum([max(0, 1-dot_product_y_test[i])  for i in range(N_test)]),
                                      'squared_hinge': np.sum([max(0, 1-dot_product_y_test[i])**2  for i in range(N_test)]) }

                    train_errors.append(dict_loss_train[type_loss])
                    test_errors.append(dict_loss_test[type_loss])
                    
                    betas_non_null.append(beta)

                       
                       
        #---RESULTS

            loss_decrease_bool = all(train_errors[i] >= train_errors[i+1] for i in range(len(train_errors)-1))
            write_and_print('LS decreases over the '+str(len(train_errors))+' different supports : '+str(loss_decrease_bool), f)

            if len(test_errors)>0:
                argmin_test_errors = np.argmin(test_errors)
                min_test_error = test_errors[argmin_test_errors]
               
                beta_min = betas_non_null[argmin_test_errors]
                beta_min_support = set(np.where(beta_min!=0)[0])
                
                #Print
                write_and_print('Minimal test error of '+str(np.round(min_test_error,3)) ,f)
                write_and_print('\nBest support : '+ str(list(sorted(beta_min_support))) ,f)
                



            #---Try if best results
                if(min_test_error < best_test_error):
                    best_beta = beta_min
                    best_beta_support = beta_min_support
                    
                    best_K0 = K0
                    best_test_error = min_test_error
        
        
    #---BEST
        write_and_print('\n\nBest K0 : '+str(best_K0)+'   Best test error : '+str(round(best_test_error,3)) ,f)
        write_and_print('Best support : '+str(list(sorted(best_beta_support))) ,f)



