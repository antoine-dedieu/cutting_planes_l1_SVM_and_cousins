import numpy as np
import time

import sys
sys.path.append('../algorithms')
from heuristics_classification import *
from Gurobi_SVM import *







#The next procedure tries local improvements on all the different supports

def MIO_improvements_real_datasets(type_loss, type_penalization, X, y, train_errors, betas_grid, K0_list_MIO, number_decreases, alpha_list, time_limit, f):    
    
#K0_list_MIO : sublist of values for MIO with CONSECUTIVE values
#alpha_list  : different for L1 and L2
#real_beta   : for before/after VS error

    
    write_and_print('\n\n------------------------------------------------------\nMIO for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    start = time.time()

    

#---RESULTS

    new_betas_grid_MIO = [ [] for K0 in K0_list_MIO]
    new_train_errors_MIO = [ [] for K0 in K0_list_MIO]


#index_support : stores index of different support
#ratio_improvements : improvement by MIO on every support

    all_model_statuts = [ [] for K0 in K0_list_MIO]
    ratio_improvements = [ [] for K0 in K0_list_MIO]

    index_support = [ [] for K0 in K0_list_MIO]
    


#---LOOP

    K0_min = K0_list_MIO[0]

    for K0 in K0_list_MIO: 

        for loop in range(number_decreases-1):
            beta_1, beta0_1 = betas_grid[K0][loop]
            beta_2, beta0_2 = betas_grid[K0][loop+1]

            support1 = np.where(beta_1!=0)[0]
            support2 = np.where(beta_2!=0)[0]
            
            
            #We look for the different support 
            #We solve a support of size K0 for its lowest alpha

            if(loop==number_decreases-2  or (len(support1)==K0 and (len(set(support1)-set(support2))!=0 or len(set(support2)-set(support1))!=0))):
                
                alpha = alpha_list[loop]
                index_support[K0-K0_min].append(loop)

            #---No model but warm-start
                beta_MIO, beta0, train_error, model_status, _ = Gurobi_SVM(type_loss, type_penalization, 'l0', X, y, K0, alpha, time_limit, 0, (beta_1, beta0_1))

                new_betas_matrix_MIO[K0-K0_min].append(np.array(beta_MIO))
                new_train_errors_MIO[K0-K0_min].append(train_error)


            #---Improvements
                ratio_improvement = 1 - train_error/float(train_errors[K0][loop])
                ratio_improvements[K0-K0_min].append(ratio_improvement)

                all_model_statuts[K0-K0_min].append(model_status)
                
                     
#---Results
    for K0 in K0_list_MIO:         
        if len(ratio_improvements[K0-K0_min])>0:   
            write_and_print('\n\nFor K0='+str(K0)+',    Number Gurobi runs ='+str(len(ratio_improvements[K0-K0_min]))+',    Max improvement in % ='+str(round(100*np.max(ratio_improvements[K0-K0_min]),4))+',    Mean improvement in % ='+str(round(100*np.mean(ratio_improvements[K0-K0_min]),4)) ,f)
            write_and_print('\nGurobi status : '+str(all_model_statuts[K0-K0_min]) ,f)

    write_and_print('\n\nTime : '+str(time.time()-start), f)
        
    return new_train_errors_MIO, new_betas_matrix_MIO, ratio_improvements, index_support, all_model_statuts






def averaged_objective_value_after_MIO(type_penalization, train_errors_RFE, train_errors_heuristic1, best_train_error_heuristic3, train_errors_heuristic3_NS, train_errors_heuristic3_Gurobi, K0_list_MIO, index_support, f):
    
#INDEX SUPPORT -> CAREFULL !!

    write_and_print('\n\n------------------------------------------------------\nAVERAGED OBJECTIVE VALUE AFTER MIO for '+type_penalization+ ' LOSS', f)


    dict_type ={1:'RFE              ',
                2:'Heuristic 1      ',
                3:'Heuristic 3-3b   ',
                4:'H3-3b + RNS      ',
                5:'H3-3b + RNS + MIO'}


    K0_min = K0_list_MIO[0]
    
    for K0 in K0_list_MIO:
        write_and_print('\n\nK0 = '+str(K0) ,f)

        idx = 0
        for train_errors in [train_errors_RFE, train_errors_heuristic1, best_train_error_heuristic3, train_errors_heuristic3_NS, train_errors_heuristic3_Gurobi]:
            idx +=1
            if(idx<5):
                #CHECK !!
                train_errors_K0 = [train_errors[K0][i] for i in index_support[K0-K0_min]]
            else:
                train_errors_K0 = train_errors[K0-K0_min]
                
            mean = round(np.mean(train_errors_K0),3)
            write_and_print('For '+str(dict_type[idx])+' : '+str(mean), f)







            

