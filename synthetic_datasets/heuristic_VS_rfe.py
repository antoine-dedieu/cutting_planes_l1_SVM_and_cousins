import numpy as np

import sys
sys.path.append('../algorithms')
from heuristics_classification import *
from aux_heuristic_VS_rfe import *


def AUC(beta, X, y):
    y =  np.array(y)
    idx_plus  = np.where(y == 1)[0]
    idx_minus = np.where(y == -1)[0]

    AUC_score = 0

    for i in idx_plus:
        for j in idx_minus:
            AUC_score += int(np.dot(beta, X[i, :] -X[j, :]) >0)

    AUC_score /= float(len(idx_plus)*len(idx_minus))
    return AUC_score


def write_and_print(text,f):
    print text
    f.write('\n'+text)




def heuristic_VS_rfe(type_loss, X_train, X_test, y_train, y_test, real_beta, betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1, K0_list, number_decreases, f):

#K0_SUBLIST : used to plot the results, based on support of size K0 -> for every penalization, K0 for which we have the maxima

#For every penalization, we want to :
#1/ Have the best K0 to plot the 3 metrics 
#2/ For every K0, have all the AUC test errors and VS errors 


    N_train,P = X_train.shape
    N_test,P = X_test.shape
    
    real_support = set(sorted(np.where(real_beta!=0)[0]))
    K0_min = K0_list[0]
    
    
    dict_loss = {'squared_hinge':'SQUARED HINGE',
                'hinge':'HINGE'}
    
    list_title = ['HEURISTIC L2','HEURISTIC L1', 'RFE L2', 'RFE L1']
    
    
    aux = -1

#---Results for PLOTS 
    train_errors = [  [[1e10] for K0 in K0_list] for i in range(4) ]
    test_errors  = [  [[1e10] for K0 in K0_list] for i in range(4) ]
    test_AUC_errors    = [ [[0] for K0 in K0_list] for i in range(4) ]
    variable_selections= [ [[0] for K0 in K0_list] for i in range(4) ]

    support_changes_plots = [[[]] for i in range(4)]


    K0_sublist  = []
    max_AUC_errors_all_penalization   = []




#---SAME LOOP FOR ALL 
    for betas in [betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1]:
        aux += 1
        betas_non_null = [[np.zeros(P)] for K0 in K0_list]

        write_and_print('\n\n------------------------------------------------------\nFOR '+str(dict_loss[type_loss])+' AND '+str(list_title[aux]) ,f)
    
    #---Best results
        max_AUC_errors = [0] #K0=0
        max_betas = [[]]
       
    
    
    #---COMPUTE
        for K0 in K0_list:
            write_and_print('\n\n\nK0 = '+str(K0), f)

        #---For choice of K0
            AUC_scores_size_K0    = [0] 
            betas_support_size_K0 = [np.zeros(P)] 
            old_support = []


            for loop in range(len(betas[K0-K0_min])):
                beta, beta0 = betas[K0-K0_min][loop]

            #---Compute errors to PLOT
                dot_product_y_train = y_train*(np.dot(X_train,beta)+ beta0*np.ones(N_train))
                dot_product_y_test = y_test*(np.dot(X_test,beta)+ beta0*np.ones(N_test))

                dict_loss_train = {'hinge': np.sum([max(0, 1-dot_product_y_train[i])  for i in range(N_train)]),
                                   'squared_hinge': np.sum([max(0, 1-dot_product_y_train[i])**2  for i in range(N_train)]) }

                dict_loss_test = {'hinge': np.sum([max(0, 1-dot_product_y_test[i])  for i in range(N_test)]),
                                  'squared_hinge': np.sum([max(0, 1-dot_product_y_test[i])**2  for i in range(N_test)]) }
            
            #---Train and test
                train_errors[aux][K0-K0_min].append(dict_loss_train[type_loss])
                test_errors[aux][K0-K0_min].append(dict_loss_test[type_loss])

            #---AUC
                AUC_score = AUC(beta, X_test, y_test)
                test_AUC_errors[aux][K0-K0_min].append(AUC_score)
                
            #---VS
                support = set(np.where(beta!=0)[0]) 
                variable_selections[aux][K0-K0_min].append(len(support-real_support) + len(real_support-support))


            #---Check is support changes for plots
                if len(set(support)-set(old_support))!=0 or len(set(old_support)-set(support))!=0 or i==number_decreases-1:
                    support_changes_plots[aux].append(loop)   
                old_support = support


            #---Condition on support to select best K0
                if len(np.where(beta!=0)[0]) == K0:
                    AUC_scores_size_K0.append(AUC_score)
                    betas_support_size_K0.append(beta)



                   

        #---RESULTS FOR K0
            loss_decrease_bool = all(train_errors[i] >= train_errors[i+1] for i in range(len(train_errors)-1))
            write_and_print('LS decreases over the '+str(len(train_errors))+' different supports : '+str(loss_decrease_bool), f)


            if len(AUC_scores_size_K0)>0:
                max_AUC_error, beta_max_support = aux_results_K0(AUC_scores_size_K0, betas_support_size_K0, real_support, f)
                max_AUC_errors.append(max_AUC_error)
                max_betas.append(beta_max_support)

                
        sorted_list, max_test_errors_penalization = aux_results_penalization(max_AUC_errors, max_betas, real_support, f)
        K0_sublist += [sorted_list[0]]
        max_AUC_errors_all_penalization.append((max_test_errors_penalization[0], sorted_list[0]))

        if aux<2:
            K0_sublist += [sorted_list[1]]
            max_AUC_errors_all_penalization.append((max_test_errors_penalization[1], sorted_list[1]))

    return K0_sublist, support_changes_plots, train_errors, test_AUC_errors, variable_selections, max_AUC_errors_all_penalization
 









def old_heuristic_VS_rfe(type_loss, X_train, X_test, y_train, y_test, real_beta, betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1, K0_list, f):

#K0_SUBLIST : USED TO PLOT THE RESULTS -> for every penalization, K0 for which we have the minima
#TEST_ERROR_LIST, GOOD_VARIABLES_LIST : USED TO SEE IF WE BEAT LASSO

#For every penalization, we want to :
#1/ Have the best K0 to plot the 3 metrics and see if we beat Lasso
#2/ For every K0, ave all the test errors and VS errors on different supports to plot and see if we beat Lasso
#3/ For every K0, see if the LS decreases, if the best support in test set is different to the last one


    N_train,P = X_train.shape
    N_test,P = X_test.shape
    
    real_support = set(np.where(real_beta!=0)[0])
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
        
        K0_min = K0_list[0]
        
        for K0 in K0_list:
            write_and_print('\n\nK0 = '+str(K0), f)
           
            train_errors = [1e10] 
            test_errors = [1e10] 
            variable_selections = [np.zeros(P)] 
            betas_non_null = [np.zeros(P)] 
                       

            for loop in range(len(betas[K0-K0_min])):
 
                beta, beta0 = betas[K0-K0_min][loop]
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
                    variable_selections.append(len(support-real_support) + len(real_support-support))
                       
                       
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
                write_and_print('Good variables : '+str(len( set(real_support) & set(beta_min_support))) ,f)  
                if len(beta_min_support)>0:
                    write_and_print('Variable selection error : '+str(  len(beta_min_support-real_support) + len(real_support-beta_min_support)  ) ,f)

                write_and_print('\nBest support : '+ str(list(sorted(beta_min_support))) ,f)
                write_and_print('Real support : '+ str(list(real_support)) ,f)



            #---Try if best results
                if(min_test_error < best_test_error):
                    best_beta = beta_min
                    best_beta_support = set(beta_min_support)
                    
                    best_K0 = K0
                    best_test_error = min_test_error
        
        
    #---BEST
        if len(best_beta_support)>0:
            VS_error = len(best_beta_support-real_support) + len(real_support-best_beta_support)
        else:
            VS_error = len(real_support)
        
        write_and_print('\n\nBest K0 : '+str(best_K0)+'   Best test error : '+str(round(best_test_error,3)) ,f)
        write_and_print('VS error : ' +str(VS_error)+'    Good variables : '+str(len( set(real_support) & set(best_beta_support))) ,f)
        write_and_print('Best support : '+str(list(sorted(best_beta_support)))+'   Real support : '+ str(list(real_support)), f)






