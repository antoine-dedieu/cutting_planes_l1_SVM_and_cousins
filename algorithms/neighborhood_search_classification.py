import numpy as np
import time

import sys

from algorithm1_classification import *
from neighborhood_search_tools import *








def randomized_NS(number_NS, type_loss, type_penalization, X, y, original_train_errors, original_test_errors, train_errors, betas_grid, K0_list, number_decreases, alpha_list, X_add, L, epsilon, time_limit, f):    


#NUMBER_NS: if odd then from k_min to k_max / if even then from k_max to k_min
#ORIGINAL_TEST_ERRORS: to compute improvements around min
    
    
    write_and_print('\n\n\n------------------------------------------------------\nROUND '+str(number_NS)+' OF RANDOMIZED NEIGHBORHOOD SEARCH for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    start = time.time()

    N,P = X.shape[0],X.shape[1]
        

#---ODD OR EVEN
    K0_max = K0_list[::-1][0]
    
    if number_NS%2==0:
        K0_list_NS = range(1,K0_max+1)
    else:
        K0_list_NS = range(1,K0_max+1)[::-1]



#---INITIALIZE
    new_train_errors = train_errors
    new_betas_grid = betas_grid

    
    ratio_improvements = [ [[] for loop in range(number_decreases)]  for K0 in K0_list]    
    move_from = [[0 for K0 in K0_list_NS] for i in range(5)]
    
    

    #Get the different supports
    matrix_different_support = Matrix_different_support(betas_grid, K0_list, number_decreases)
    
        
    
#---MAIN LOOP
    #CAREFULL WITH K0_MAX
    
    idx = -1
    for K0 in K0_list_NS:
        idx += 1


        for loop in range(number_decreases):
            
            #Current value
            beta_0, beta0_0 = new_betas_grid[K0][loop]
            train_error_0 = train_errors[K0][loop]
            
            #ALPHA TO CHANGE
            previous_index,next_index = matrix_different_support[K0][loop]
            alpha = alpha_list[loop]
            
            
        #---Change K
            
            #K0-1
            start1, start0_1 = new_betas_grid[K0-1][loop]
            start1 = shuffle_half_support(start1, K0, P)
            beta_start1 = start0_1*np.ones(P+1)
            beta_start1[:P] = start1
            
            beta_1, beta0_1, train_error_1 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start1, X_add, L, epsilon, time_limit)   
            
            
            #K0+1
            start3, start0_3 = new_betas_grid[min(K0+1,K0_max)][loop]
            start3 = shuffle_half_support(start3, K0, P)
            beta_start3 = start0_3*np.ones(P+1)
            beta_start3[:P] = start3
            
            beta_3, beta0_3, train_error_3 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start3, X_add, L, epsilon, time_limit)   
            

            
        #---Change alpha
            start2, start0_2 = new_betas_grid[K0][previous_index]
            start2 = shuffle_half_support(start2, K0, P)
            beta_start2 = start0_2*np.ones(P+1)
            beta_start2[:P] = start2
            
            beta_2, beta0_2, train_error_2 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start2, X_add, L, epsilon, time_limit)   
            
            
            #If nothing changes till the last gamma, next_index=0 -> we shuffle the actual support
            if(next_index!=0):
                start4, start0_4 = new_betas_grid[K0][next_index]
            else:
                start4, start0_4 = new_betas_grid[K0][loop]
                
            start4 = shuffle_half_support(start4, K0, P)
            beta_start4 = start0_4*np.ones(P+1)
            beta_start4[:P] = start4
            
            beta_4, beta0_4, train_error_4 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start4, X_add, L, epsilon, time_limit)   
            
            
            
            
        #---Keep the best beta
            neighbors_train_errors = [train_error_0, train_error_1, train_error_2, train_error_3, train_error_4]
            neighbors_betas = [(beta_0, beta0_0), (beta_1, beta0_1), (beta_2, beta0_2), (beta_3, beta0_3), (beta_4, beta0_4)]

            argmin = np.argmin(neighbors_train_errors)
            move_from[argmin][idx]+=1 #tricky
            
            new_train_errors[K0][loop] = neighbors_train_errors[argmin]
            new_betas_grid[K0][loop] = neighbors_betas[argmin]
            


        #---Improvements
            original_train_error = original_train_errors[K0][loop]
            
            if(original_train_error>0):
                ratio_improvements[K0][loop] = np.round(1 - neighbors_train_errors[argmin]/float(original_train_error),6)
            else:
                ratio_improvements[K0][loop] = 0
    
    
    
    #---IMPROVEMENTS AROUND THE MINIMUM -> 5 values
        if len(original_test_errors)>0:
            
            test_argmin = np.argmin(original_test_errors[K0])
            ratio_improvements_around_min = []
            
            for i in range(max(test_argmin-2,0) , min(test_argmin+2,number_decreases)):
                ratio_improvements_around_min.append(ratio_improvements[K0][i])
        
        else:
            ratio_improvements_around_min = [0]

        
        
        #STATISTICS ON THE IMPROVEMENT
        write_and_print('\nFor K0='+str(K0)+',    Max improvement in % = '+str(round(100*np.max(ratio_improvements[K0]),4))+',    Mean improvement in % = '+str(round(100*np.mean(ratio_improvements[K0]),4))+',    Mean improvement around min in % = '+str(round(100*np.mean(ratio_improvements_around_min),4))  ,f)



    #UNDERSTAND THE TREND
    write_and_print('\n\nStay             : '+str(move_from[0]), f)
    write_and_print('\nMove from K0-1   : '+str(move_from[1]), f)
    write_and_print('\nMove from alpha-1  : '+str(move_from[2]), f)
    write_and_print('\nMove from K0+1   : '+str(move_from[3]), f)
    write_and_print('\nMove from alpha+1   : '+str(move_from[4]), f)

    write_and_print('\nTime : '+str(time.time()-start), f)

    return new_train_errors, new_betas_grid, ratio_improvements










def deterministic_NS(number_NS, type_loss, type_penalization, X, y, original_train_errors, original_test_errors, train_errors, betas_grid, K0_list, number_decreases, alpha_list, X_add, L, epsilon, time_limit, f):    


#NUMBER_NS: if odd then from k_min to k_max / if even then from k_max to k_min
#ORIGINAL_TEST_ERRORS: to compute improvements around min
    
    
    write_and_print('\n\n\n------------------------------------------------------\nROUND '+str(number_NS)+' OF DETERMINISTIC NEIGHBORHOOD SEARCH for '+str(type_loss)+' loss and '+str(type_penalization)+ ' penalization: ',f)
    start = time.time()

    N,P = X.shape[0],X.shape[1]
        

#---ODD OR EVEN
    K0_max = K0_list[::-1][0]
    
    if number_NS%2==0:
        K0_list_NS = range(1,K0_max+1)
    else:
        K0_list_NS = range(1,K0_max+1)[::-1]



#---INITIALIZE
    new_train_errors = train_errors
    new_betas_grid = betas_grid

    
    ratio_improvements = [ [[] for loop in range(number_decreases)]  for K0 in K0_list]    
    move_from = [[0 for K0 in K0_list_NS] for i in range(5)]
    
    

    #Get the different supports
    matrix_different_support = Matrix_different_support(betas_grid, K0_list, number_decreases)
    
        
    
#---MAIN LOOP
    #CAREFULL WITH K0_MAX
    
    idx = -1
    for K0 in K0_list_NS:
        idx += 1


        for loop in range(number_decreases):
            
            #Current value
            beta_0, beta0_0 = new_betas_grid[K0][loop]
            train_error_0 = train_errors[K0][loop]
            
            #ALPHA TO CHANGE
            previous_index,next_index = matrix_different_support[K0][loop]
            alpha = alpha_list[loop]
            
            
        #---Change K
            
            #K0-1
            start1, start0_1 = new_betas_grid[K0-1][loop]
            beta_start1 = start0_1*np.ones(P+1)
            beta_start1[:P] = start1
            
            beta_1, beta0_1, train_error_1 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start1, X_add, L, epsilon, time_limit)   
            
            
            #K0+1
            start3, start0_3 = new_betas_grid[min(K0+1,K0_max)][loop]
            beta_start3 = start0_3*np.ones(P+1)
            beta_start3[:P] = start3
            
            beta_3, beta0_3, train_error_3 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start3, X_add, L, epsilon, time_limit)   
            

            
        #---Change alpha
            start2, start0_2 = new_betas_grid[K0][previous_index]
            beta_start2 = start0_2*np.ones(P+1)
            beta_start2[:P] = start2
            
            beta_2, beta0_2, train_error_2 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start2, X_add, L, epsilon, time_limit)   
            
            
            #If nothing changes till the last gamma, next_index=0 -> we shuffle the actual support
            if(next_index!=0):
                start4, start0_4 = new_betas_grid[K0][next_index]
            else:
                start4, start0_4 = new_betas_grid[K0][loop]
                
            beta_start4 = start0_4*np.ones(P+1)
            beta_start4[:P] = start4
            
            beta_4, beta0_4, train_error_4 = algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, beta_start4, X_add, L, epsilon, time_limit)   
            
            
            
            
        #---Keep the best beta
            neighbors_train_errors = [train_error_0, train_error_1, train_error_2, train_error_3, train_error_4]
            neighbors_betas = [(beta_0, beta0_0), (beta_1, beta0_1), (beta_2, beta0_2), (beta_3, beta0_3), (beta_4, beta0_4)]

            argmin = np.argmin(neighbors_train_errors)
            move_from[argmin][idx]+=1 #tricky
            
            new_train_errors[K0][loop] = neighbors_train_errors[argmin]
            new_betas_grid[K0][loop] = neighbors_betas[argmin]
            


        #---Improvements
            original_train_error = original_train_errors[K0][loop]
            
            if(original_train_error>0):
                ratio_improvements[K0][loop] = np.round(1 - neighbors_train_errors[argmin]/float(original_train_error),6)
            else:
                ratio_improvements[K0][loop] = 0
    
    
    
    #---IMPROVEMENTS AROUND THE MINIMUM -> 5 values
        if len(original_test_errors)>0:
            
            test_argmin = np.argmin(original_test_errors[K0])
            ratio_improvements_around_min = []
            
            for i in range(max(test_argmin-2,0) , min(test_argmin+2,number_decreases)):
                ratio_improvements_around_min.append(ratio_improvements[K0][i])
        
        else:
            ratio_improvements_around_min = [0]

        
        
        #STATISTICS ON THE IMPROVEMENT
        write_and_print('\nFor K0='+str(K0)+',    Max improvement in % = '+str(round(100*np.max(ratio_improvements[K0]),4))+',    Mean improvement in % = '+str(round(100*np.mean(ratio_improvements[K0]),4))+',    Mean improvement around min in % = '+str(round(100*np.mean(ratio_improvements_around_min),4))  ,f)



    #UNDERSTAND THE TREND
    write_and_print('\n\nStay             : '+str(move_from[0]), f)
    write_and_print('\nMove from K0-1   : '+str(move_from[1]), f)
    write_and_print('\nMove from alpha-1  : '+str(move_from[2]), f)
    write_and_print('\nMove from K0+1   : '+str(move_from[3]), f)
    write_and_print('\nMove from alpha+1   : '+str(move_from[4]), f)

    write_and_print('\nTime : '+str(time.time()-start), f)

    return new_train_errors, new_betas_grid, ratio_improvements















