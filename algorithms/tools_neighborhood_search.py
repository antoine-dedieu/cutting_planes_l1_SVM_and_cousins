import numpy as np

from heuristics_classification import *



def shuffle_half_support(current_beta, K0, P ):
    
    beta    = np.copy(current_beta)
    support = np.where(beta!=0)[0]
    np.random.shuffle(support)

    beta[np.random.randint(P,size=int((K0+1)/2))] = beta[support[:int((K0+1)/2)]]
    beta[support[:int((K0+1)/2)]] = np.zeros(int((K0+1)/2))

    return beta






def matrix_different_support(betas_list, K0_list, N_alpha):

#FOR EVERY POINT I THE GRID, RETURN THE POINT ON THE RIGHT AND ON THE LEFT WITH DIFFERENT SUPPORT 

    matrix_different_support=[[] for K0 in K0_list]


    for K0 in K0_list: 
        previous_index = 0
        support1 = np.where(betas_list[K0][0][0]!=0)[0]
        support2 = np.where(betas_list[K0][0][0]!=0)[0]


        next_index=0
        while(len(set(support1)-set(support2))==0 and len(set(support2)-set(support1))==0 and next_index+1<N_alpha):
            next_index += 1
            support2    = np.where(betas_grid[K0][next_index][0]!=0)[0]
        matrix_different_support[K0].append((previous_index,next_index%(N_alpha-1)))


        #LOOP
        for loop in range(1,N_alpha):
            if(loop<next_index):
                matrix_different_support[K0].append((previous_index,next_index%(N_alpha-1)))

            #loop=number_decreases
            elif(loop+1==N_alpha):
                matrix_different_support[K0].append((previous_index,0))
                
            else:
                previous_index = loop-1
                support1 = np.where(betas_list[K0][loop][0]!=0)[0]
                support2 = np.where(betas_list[K0][loop][0]!=0)[0]

                next_index=loop
                while(len(set(support1)-set(support2))==0 and len(set(support2)-set(support1))==0 and next_index+1<N_alpha):
                    next_index += 1
                    support2 = np.where(betas_list[K0][next_index][0]!=0)[0]
                
                matrix_different_support[K0].append((previous_index,next_index%(N_alpha-1)))
            
        

    return matrix_different_support









def averaged_objective_value_after_NS(type_penalization, train_errors_RFE, train_errors_heuristic1, best_train_error_heuristic2, best_train_error_heuristic3, train_errors_NS_randomized, train_errors_NS_deterministic, K0_list, number_decreases, f):
    
  
    write_and_print('\n\nAVERAGED OBJECTIVE VALUE AFTER NS for '+type_penalization+ ' LOSS', f)
    

    dict_type ={1:'RFE                     ',
                2:'Heuristic 1             ',
                3:'Heuristic 2-2b          ',
                4:'Heuristic 3-3b          ',
                5:'H3-3b + randomized    NS',
                6:'H3-3b + deterministic NS'}

    for K0 in K0_list[1:]:
        write_and_print('\n\nK0 = '+str(K0) ,f)

        idx = 0
        for train_errors in [train_errors_RFE, train_errors_heuristic1, best_train_error_heuristic2, best_train_error_heuristic3, train_errors_NS_randomized, train_errors_NS_deterministic]:
            idx +=1
            mean = round(np.mean(train_errors[K0]),3)
            write_and_print('For '+str(dict_type[idx])+' : '+str(mean), f)



