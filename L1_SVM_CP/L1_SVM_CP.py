import numpy as np
from gurobipy import *
from L1_SVM_CP_model import *

import time
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------INITIALIZE WITH RFE--------------------------------------------------


def init_CP_clustering(X_train, y_train, n_samples, f):

####furthest of closest to mid point ?
####independent of alpha ?



#OUTPUT
#index_CP: index of constraints to whihc we start
    
    start = time.time()


    X_plus = X_train[y_train==1]
    mean_plus = np.mean(X_plus, axis=0)

    X_minus = X_train[y_train==-1]
    mean_minus = np.mean(X_minus, axis=0)


#---Hyperplan between two centroids
    vect_orth_hyperplan = mean_plus - mean_minus
    point_hyperplan     = 0.5*(mean_plus + mean_minus)
    b0 = -np.dot(point_hyperplan, vect_orth_hyperplan)


#---Rank point according to distance to one mean or to distance to hyperplan .??


#--- +1 class

    #dist_X_plus   = [np.dot(X_plus[i,:] - mean_plus, -vect_orth_hyperplan) + b0 for i in range(X_plus.shape[0])]
    #index_CP_plus = np.array(dist_X_plus).argsort()[::-1][:n_samples/2]
    #X_init_plus   = np.matrix([X_plus[i,:] for i in index_CP_plus])

    dist_X_plus   = [np.dot(X_plus[i,:] - point_hyperplan, vect_orth_hyperplan) + b0 for i in range(X_plus.shape[0])]
    index_CP_plus = np.array(np.abs(dist_X_plus)).argsort()[:n_samples/2]
    X_init_plus   = np.matrix([X_plus[i,:] for i in index_CP_plus])


#--- -1 class
    
    #dist_X_minus   = [np.dot(X_minus[i,:] - mean_minus, vect_orth_hyperplan) + b0 for i in range(X_minus.shape[0])]
    #index_CP_minus = np.array(dist_X_minus).argsort()[::-1][:n_samples/2]
    #X_init_minus    = np.matrix([X_minus[i,:] for i in index_CP_minus])

    dist_X_minus    = [np.dot(X_minus[i,:] - point_hyperplan, vect_orth_hyperplan) + b0 for i in range(X_minus.shape[0])]
    index_CP_minus  = np.array(np.abs(dist_X_minus)).argsort()[:n_samples/2]
    X_init_minus    = np.matrix([X_minus[i,:] for i in index_CP_minus])
    


    index_CP = np.concatenate([index_CP_plus, index_CP_minus])
    write_and_print('Time init: '+str(time.time()-start), f)

    return index_CP




def init_CP_random_sampling(X_train, y_train, n_samples, f):

    N0 = 25
    time_limit = 30

    start = time.time()
    count_score_lines = np.zeros(N)

    for k in range(100):
        subset = np.sort(random.sample(xrange(N),N0))
        X_train_reduced = np.array([X_train[i,:] for i in subset])
        y_train_reduced = np.array([y_train[i]   for i in subset])

        model = L1_SVM_CP_model(X_train_reduced, y_train_reduced, range(N0), alpha, time_limit, 0) 
        model.optimize()

        dual_slack  = [model.getConstrByName('slack_'+str(i)).Pi for i in range(N0)]
        #print np.sum(dual_slack)


        for i in range(N0):
            count_score_lines[subset[i]] += (0 < dual_slack[i])

            
    print 'Time: '+str(time.time() - start)

    index_CP = np.arange(N)[count_score_lines>=1].tolist()

    return index_CP




def init_CP_norm_samples(X_train, y_train, n_samples, f):

    sum_lines = np.abs(np.sum(X_train, axis=1))
    argsort_lines = np.argsort(sum_lines)
    index_CP      = argsort_lines[::-1][:n_samples].tolist()
    
    return index_CP







def L1_SVM_CP(X_train, y_train, index_CP, alpha, epsilon_RC, time_limit, model, delete_constraints, f):


#INPUT
#n_features_RFE: number of features to give to RFE to intialize
#epsilon_RC    : maximum non negatuve reduced cost

    N,P = X_train.shape
    aux = 0   #count the number of rounds 
    is_L1_SVM = (len(index_CP) == N)
    
    #index_CG = index_CG.tolist()


#---Build the model
    start = time.time()
    model = L1_SVM_CP_model(X_train, y_train, index_CP, alpha, time_limit, model, f) #model=0 -> no warm start
    


#---Infinite loop until all the constraints are satisfied
    while True: 
        aux += 1
        
    #---Solve the relax problem to get the dual solution
        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)

        
        
    #---Compute all reduce cost and look for variable with negative reduced costs
        beta_plus  = np.array([model.getVarByName('beta_+_'+str(i)).X for i in range(P)])
        beta_minus = np.array([model.getVarByName('beta_-_'+str(i)).X  for i in range(P)])
        beta       = beta_plus - beta_minus
        b0         = model.getVarByName('b0').X
        
        violated_constraints     = []
        most_violated_constraint = -1
        most_violated_cost       = 0
        
        
        for constraint in set(range(N))-set(index_CP):
            constraint_value = 1 - y_train[constraint]*(np.dot(X_train[constraint], beta) + b0)
            if constraint_value > epsilon_RC:
                violated_constraints.append(constraint)

            #if constraint_value>most_violated_cost:
            #    most_violated_constraint = constraint
            #    most_violated_cost = constraint_value
                
        
        
    #---Add the column with the most most violated constraint to the original model (not the relax !!)
        n_constraints_to_add = len(violated_constraints)

        if n_constraints_to_add>0:
            write_and_print('Number of constraints added: '+str(n_constraints_to_add), f)
            model = add_constraints_L1_SVM(X_train, y_train, model, violated_constraints, range(P)) 
            model.update()

            for violated_constraint in violated_constraints:
                index_CP.append(violated_constraint)

        #if most_violated_constraint >= 0:
        #    model = add_constraints_L1_SVM(X_train, y_train, model, [most_violated_constraint]) 
        #    model.update()
        #    index_CP.append(most_violated_constraint)

        
    #---WE ALWAYS BREAK FOR L1 SVM     
        else:
            break



#---Solution

    write_and_print('Number of rounds: '+str(aux), f)

    beta_plus  = np.array([model.getVarByName('beta_+_'+str(i)).X for i in range(P)])
    beta_minus = np.array([model.getVarByName('beta_-_'+str(i)).X for i in range(P)])
    beta       = np.round(np.array(beta_plus) - np.array(beta_minus),6)
    b0         = model.getVarByName('b0').X
    support = np.where(beta!=0)[0]

    write_and_print('\nObj value   = '+str(model.ObjVal), f)
    write_and_print('\nLen support       = '+str(len(support)), f)

    solution_dual = np.array([model.getConstrByName('slack_'+str(index)).Pi for index in index_CP])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)


#---IF DELETE APPROACH, then delete the constraint not in the dual solution

    if delete_constraints and not is_L1_SVM: #CANNOT HAPPEN FOR L1 SVM

        idx_to_removes   = np.array(index_CP)[solution_dual==0] #non in dual solution
        slacks_to_remove = np.array([model.getConstrByName('slack_'+str(index)) for index in idx_to_removes ]) 

        for idx_to_remove in idx_to_removes:
            index_CP.remove(idx_to_remove)

        for slack_to_remove in slacks_to_remove:
            model.remove(slack_to_remove)
        model.update()
        
    
    time_CP = time.time()-start    
    write_and_print('Time = '+str(time_CP), f)

    return beta, support, time_CP, model, index_CP, model.ObjVal









