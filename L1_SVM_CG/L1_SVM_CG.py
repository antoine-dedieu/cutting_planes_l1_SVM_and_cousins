import numpy as np
from gurobipy import *
from L1_SVM_CG_model import *

from scipy.stats.stats import pearsonr 

import time
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------INITIALIZE WITH RFE--------------------------------------------------


def RFE_for_CG(type_RFE, X_train, y_train, alpha, n_features, f):

#Run a two stpe RFE as suggested

#INPUT
#type_RFE = hinge_l2         -> hinge+l2 (only on dual)
#type_RFE = squared_hinge_l1 -> squared_hinge+l2 (only on primal)

#OUTPUT
#index_CG the
    
    start = time.time()
    N,P = X_train.shape

#---First RFE by removing half of the features at every iteration
    if type_RFE== 'hinge_l2':
        estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=1/(2*float(alpha)))
    elif type_RFE== 'squared_hinge_l1':
        estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(alpha))

    selector = RFE(estimator, n_features_to_select=100, step=0.5)
    selector = selector.fit(X_train, y_train)


#---Compute the reduced data
    support_RFE_first_step = np.where(selector.support_==True)[0]
    X_train_reduced = []

    for i in range(N):
        X_train_reduced.append(X_train[i,:][support_RFE_first_step])


#---Second RFE removing one feature at a time
    if type_RFE== 'hinge_l2':
        estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=1/(2*float(alpha)))
    elif type_RFE== 'squared_hinge_l1':
        estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(alpha))

    selector = RFE(estimator, n_features_to_select=n_features, step=0.9)
    selector = selector.fit(X_train_reduced, y_train)

    support_RFE_second_step = np.where(selector.ranking_ < 2)[0].tolist()
    index_RFE = support_RFE_first_step[support_RFE_second_step]

    time_RFE = time.time()-start

    write_and_print('Time RFE for column subset selection: '+str(time_RFE), f)

    return index_RFE.tolist(), time_RFE




def liblinear_for_CG(type_liblinear, X, y, alpha, f):

    start = time.time()
    N = X.shape[0]

    if type_liblinear== 'hinge_l2':
        estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=1/(2*float(alpha)))
    elif type_liblinear== 'squared_hinge_l1':
        estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(alpha))

    estimator = estimator.fit(X, y)

    beta_liblinear, b0 = estimator.coef_[0], estimator.intercept_
    support            = np.where(beta_liblinear !=0)[0].tolist()
    beta_liblinear     = beta_liblinear[support]


    write_and_print('Len support liblinear: '+str(len(support)), f)

    time_liblinear = time.time()-start
    write_and_print('Time liblinear for '+type_liblinear+': '+str(time_liblinear), f)

    return [beta_liblinear, b0], support, time_liblinear






def init_correlation(X_train, y_train, n_features, f):

#Run a two stpe RFE as suggested

#OUTPUT
#index_CG the
    
    start = time.time()
    N,P = X_train.shape

#---First RFE by removing half of the features at every iteration
    if(n_features<=P):
        correlations    = np.dot(X_train.T, y_train)       #class are balanced so we always can compute this way
        argsort_columns = np.argsort(np.abs(correlations))
        index_CG        = argsort_columns[::-1][:n_features]

    time_correl = time.time()-start
    write_and_print('Time correlation for column subset selection: '+str(time_correl), f)
    return index_CG.tolist(), time_correl






def L1_SVM_CG(X_train, y_train, index_CG, alpha, epsilon_RC, time_limit, model, warm_start, delete_columns, f):


#INPUT
#n_features_RFE : number of features to give to RFE to intialize
#epsilon_RC     : maximum non negatuve reduced cost
#delete_columns : boolean indicating whether we have to delete the columns not in the support
    
    start = time.time()
    
    N,P = X_train.shape
    aux = 0   #count he number of rounds 
    #index_CG = index_CG.tolist()


#---Build the model
    model     = L1_SVM_CG_model(X_train, y_train, index_CG, alpha, time_limit, model, warm_start, f) #model=0 -> no warm start else update the objective function
    is_L1_SVM = (len(index_CG) == P)

    columns_to_check = list(set(range(P))-set(index_CG))
    ones_P = np.ones(P)
    

#---Infinite loop until all the variables have non reduced cost
    while True:
        aux += 1
        write_and_print('Round '+str(aux), f)
        
    #---Solve the problem to get the dual solution
        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)

        if not is_L1_SVM:

        #---Compute all reduce cost and look for variable with negative reduced costs
            dual_slacks      = [model.getConstrByName('slack_'+str(i)) for i in range(N)]
            dual_slack_values= [dual_slack.Pi for dual_slack in dual_slacks]
            
            RC_aux           = np.array([y_train[i]*dual_slack_values[i] for i in range(N)])
            RC_array         = alpha*ones_P[len(columns_to_check)] - np.abs( np.dot(X_train[:, np.array(columns_to_check)].T, RC_aux) )

            violated_columns = np.array(columns_to_check)[RC_array < -epsilon_RC]
                       
            
        #---Add the column with negative reduced costs
            n_columns_to_add = violated_columns.shape[0]

            if n_columns_to_add>0:
                write_and_print('Number of columns added: '+str(n_columns_to_add), f)
                model = add_columns_L1_SVM(X_train, y_train, model, violated_columns, range(N), dual_slacks, alpha) 

                for i in range(n_columns_to_add):
                    column_to_add = violated_columns[i]
                    index_CG.append(column_to_add)
                    columns_to_check.remove(column_to_add)

            else:
                break

            
    #---WE ALWAYS BREAK FOR L1 SVM   
        else:
            break 



#---Solution
    beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in index_CG])
    beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in index_CG])
    beta    = np.array(beta_plus) - np.array(beta_minus)

    obj_val = model.ObjVal


#---TIME STOPS HERE
    time_CG = time.time()-start 
    write_and_print('\nTIME CG = '+str(time_CG), f)   
    


#---support and Objective value
    support = np.where(beta!=0)[0]
    beta    = beta[support]

    support = np.array(index_CG)[support]
    write_and_print('\nObj value   = '+str(obj_val), f)
    write_and_print('Len support = '+str(len(support)), f)

    b0   = model.getVarByName('b0').X 
    write_and_print('b0   = '+str(b0), f)


#---Violated constraints and dual support
    constraints = np.ones(N) - y_train*( np.dot(X_train[:, support], beta) + b0*np.ones(N))
    violated_constraints = np.arange(N)[constraints >= 0]
    write_and_print('\nNumber violated constraints =  '+str(violated_constraints.shape[0]), f)


    solution_dual = np.array([model.getConstrByName('slack_'+str(index)).Pi for index in range(N)])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)


#---IF DELETE APPROACH, then remove non basic features
    #if delete_columns and not is_L1_SVM: #CANNOT HAPPEN FOR L1 SVM
    #    idx_to_removes = list(set(index_CG) - set(support))

    #    betas_to_remove = np.array([model.getVarByName(name="beta_+_"+str(i)) for i in idx_to_removes] + 
    #                               [model.getVarByName(name="beta_-_"+str(i)) for i in idx_to_removes])
    #    for idx_to_remove in idx_to_removes:
    #        index_CG.remove(idx_to_remove)

    #    for beta_to_remove in betas_to_remove:
    #        model.remove(beta_to_remove)
    #    model.update()


    return [beta, b0], support, time_CG, model, index_CG, obj_val









