import numpy as np
from gurobipy import *

import time

from L1_SVM_both_CG_CP_model import *

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *

sys.path.append('../L1_SVM_CG')
from L1_SVM_CG_model import *

sys.path.append('../L1_SVM_CP')
from L1_SVM_CP_model import *





def liblinear_for_both_CG_CP(type_liblinear, X, y, alpha, f):

#Run a fast SVM on whole dataset to give a good idea for the initila set of columns and constraints

#OUTPUT
#support:       support of the SVM -> serve as intialization for columns
#idx_liblinear: constraints violated by the l1 SVM -> serve as intialization for samples

    start = time.time()
    N = X.shape[0]

    if type_liblinear== 'hinge_l2':
        estimator = svm.LinearSVC(penalty='l2', loss= 'hinge', dual=True, C=1/(2*float(alpha)))
    elif type_liblinear== 'squared_hinge_l1':
        estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(alpha))

    estimator = estimator.fit(X, y)

#---Features
    beta_liblinear, b0 = estimator.coef_[0], estimator.intercept_[0]
    support            = np.where(beta_liblinear !=0)[0].tolist()
    beta_liblinear_supp= beta_liblinear[support]
    write_and_print('Len support liblinear: '+str(len(support)), f)


#---Constraints
    constraints = np.array([1 - y[i]*( np.dot(X[i][support], beta_liblinear_supp) +b0) for i in range(N)])
    idx_liblinear = np.arange(N)[constraints >= 0].tolist()
    write_and_print('Len dual liblinear: '+str(len(idx_liblinear)), f)


    time_liblinear = time.time()-start
    write_and_print('Time liblinear for '+type_liblinear+': '+str(time_liblinear), f)
    return idx_liblinear, support, time_liblinear, beta_liblinear






def L1_SVM_both_CG_CP(X_train, y_train, index_samples, index_columns, alpha, epsilon_RC, time_limit, model, warm_start, delete_samples, f):

#START WITH A REASONABLY GOOD MODEL AND CHECK IF COLUMNS OR CONSTRAINTS ARE VIOLATED


#INPUT
#n_features_RFE : number of features to give to RFE to intialize
#epsilon_RC     : maximum non negatuve reduced cost
    
    start = time.time()
   
    N,P   = X_train.shape
    N_CP  = len(index_samples)
    P_CG  = len(index_columns)

    aux = 0   #count he number of rounds 


#---Build the model
    model = L1_SVM_both_CG_CP_model(X_train, y_train, index_samples, index_columns, alpha, time_limit, model, warm_start, f) #model=0 -> start with all samples but small subset of collumns 
                                                                         #else update the objective function
    is_L1_SVM = (N_CP == N) and (P_CG == P)
    
    beta_plus  = np.zeros(P) 
    beta_minus = np.zeros(P) 

    columns_to_check    = list(set(range(P))-set(index_columns))
    constraint_to_check = list(set(range(N))-set(index_samples))

    ones_P = np.ones(P)
    ones_N = np.ones(N)



#---Infinite loop until all the variables AND constraints have non reduced cost 
    continue_loop = True

    while continue_loop:
        aux += 1
        continue_loop = False
        write_and_print('Round '+str(aux), f)

        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)
        





        if not is_L1_SVM:

        #---Model
            dual_slacks       = [model.getConstrByName('slack_'+str(idx)) for idx in index_samples]
            dual_slack_values = [dual_slack.Pi for dual_slack in dual_slacks]

            betas_plus        = [model.getVarByName('beta_+_'+str(idx)) for idx in index_columns]
            betas_minus       = [model.getVarByName('beta_-_'+str(idx)) for idx in index_columns]
            beta              = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])
            
            b0                = model.getVarByName('b0')
            b0_value          = b0.X 


    #-------REDUCED COSTS FOR VARIABLES
        #---Look for variables with negative reduced costs
            RC_aux           = np.array([y_train[index_samples[i]]*dual_slack_values[i] for i in range(N_CP)])
            X_reduced        = X_train[np.array(index_samples),:][:,np.array(columns_to_check)]
            RC_array         = alpha*ones_P[len(columns_to_check)] - np.abs( np.dot(X_reduced.T, RC_aux) )
            violated_columns = np.array(columns_to_check)[RC_array < -epsilon_RC]
    

    #-------REDUCED COSTS FOR CONSTRAINTS
        #---Look for constraints with negative reduced costs
            X_reduced            = X_train[:, np.array(index_columns)][np.array(constraint_to_check), :]
            RC_aux               = np.dot(X_reduced, beta) + b0_value*ones_N[:N-N_CP]
            RC_array             = ones_N[:N-N_CP] - y_train[np.array(constraint_to_check)]*RC_aux
            violated_constraints = np.array(constraint_to_check)[RC_array > epsilon_RC]

                 
    #-------ADD VARIABLES    
        #---Add the columns with negative reduced costs
            n_columns_to_add = violated_columns.shape[0]
            P_CG += n_columns_to_add

            if n_columns_to_add>0:
                continue_loop = True
                
                write_and_print('Number of columns added: '+str(n_columns_to_add), f)
                write_and_print('Max violated column    : '+str(round(np.min(RC_array), 3)), f)

                model = add_columns_L1_SVM(X_train, y_train, model, violated_columns, index_samples, dual_slacks, alpha) 

                for violated_column in violated_columns:
                    index_columns.append(violated_column)
                    columns_to_check.remove(violated_column)

                    betas_plus.append(model.getVarByName('beta_+_'+str(violated_column)))
                    betas_minus.append(model.getVarByName('beta_-_'+str(violated_column)))

                write_and_print('Time adding columns = '+str(time.time()-start), f)

            
    #-------ADD CONSTRAINTS
        #---Add the constraints with negative reduced costs
            n_constraints_to_add = violated_constraints.shape[0]
            N_CP += n_constraints_to_add

            if n_constraints_to_add>0:
                continue_loop = True
                
                write_and_print('Number of constraints added: '+str(n_constraints_to_add), f)
                write_and_print('Max violated constraint    : '+str(round(np.max(RC_array), 3)), f)

                model = add_constraints_L1_SVM(X_train, y_train, model, violated_constraints, betas_plus, betas_minus, b0, index_columns) 

                for violated_constraint in violated_constraints:
                    index_samples.append(violated_constraint)
                    constraint_to_check.remove(violated_constraint)





#---Solution
    try: 
        beta_plus   = np.array([model.getVarByName('beta_+_'+str(idx)).X  for idx in index_columns])
        beta_minus  = np.array([model.getVarByName('beta_-_'+str(idx)).X  for idx in index_columns])
        beta    = np.round(np.array(beta_plus) - np.array(beta_minus),6)
    except:
        beta = np.zeros(len(index_columns))
    
    obj_val = model.ObjVal

#---TIME STOPS HERE
    time_both_CG_CP = time.time()-start 
    write_and_print('\nTIME CG-CP = '+str(time_both_CG_CP), f)


#---support and Objective value
    support = np.where(beta!=0)[0]
    write_and_print('\nObj value   = '+str(obj_val), f)
    write_and_print('Len support = '+str(len(support)), f)

    b0 = model.getVarByName('b0').X 
    write_and_print('b0   = '+str(b0), f)


#---Violated constraints and dual support
    constraints = np.ones(N_CP) - y_train[np.array(index_samples)]*( np.dot(X_train[np.array(index_samples), :][:, np.array(index_columns)], beta) + b0*np.ones(N_CP))
    violated_constraints = np.arange(N_CP)[constraints >= 0]
    write_and_print('\nNumber violated constraints =  '+str(violated_constraints.shape[0]), f)


    solution_dual = np.array([model.getConstrByName('slack_'+str(idx)).Pi for idx in index_samples])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)

    
    return beta, support, time_both_CG_CP, model, index_samples, index_columns, obj_val









