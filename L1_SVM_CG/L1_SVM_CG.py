import numpy as np
from gurobipy import *
from L1_SVM_CG_model import *

from scipy.stats.stats import pearsonr 

import time
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


#-----------------------------------------------INITIALIZE WITH RFE--------------------------------------------------


def RFE_for_CG(X_train, y_train, alpha, n_features, f):

#Run a two stpe RFE as suggested

#OUTPUT
#index_CG the
    
    start = time.time()
    N,P = X_train.shape

#---First RFE by removing half of the features at every iteration
    estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(2*alpha))
    selector = RFE(estimator, n_features_to_select=100, step=0.5)
    selector = selector.fit(X_train, y_train)


#---Compute the reduced data
    support_RFE_first_step = np.where(selector.support_==True)[0]
    X_train_reduced = []

    for i in range(N):
        X_train_reduced.append(X_train[i,:][support_RFE_first_step])


#---Second RFE removing one feature at a time
    estimator = svm.LinearSVC(penalty='l1', loss= 'squared_hinge', dual=False, C=1/float(2*alpha))
    selector = RFE(estimator, n_features_to_select=n_features, step=0.9)
    selector = selector.fit(X_train_reduced, y_train)

    support_RFE_second_step = np.where(selector.ranking_ < 2)[0].tolist()
    index_CG = support_RFE_first_step[support_RFE_second_step]

    write_and_print('Time RFE for column subset selection: '+str(time.time()-start), f)

    return index_CG



def init_correlation(X_train, y_train, n_features, f):

#Run a two stpe RFE as suggested

#OUTPUT
#index_CG the
    
    start = time.time()
    N,P = X_train.shape

#---First RFE by removing half of the features at every iteration
    if(n_features<=P):
        correlations    = np.dot(X_train.T, y_train)/np.linalg.norm(y_train) #class are balanced so we always can compute this way
        argsort_columns = np.argsort(np.abs(correlations))
        index_CG        = argsort_columns[::-1][:n_features]


    write_and_print('Time correlation for column subset selection: '+str(time.time()-start), f)
    return index_CG.tolist()






def L1_SVM_CG(X_train, y_train, index_CG, alpha, epsilon_RC, time_limit, model, delete_columns, f):


#INPUT
#n_features_RFE : number of features to give to RFE to intialize
#epsilon_RC     : maximum non negatuve reduced cost
#delete_columns : boolean indicating whether we have to delete the columns not in the support

    N,P = X_train.shape
    aux = 0   #count he number of rounds 
    #index_CG = index_CG.tolist()


#---Build the model
    start = time.time()
    model = L1_SVM_CG_model(X_train, y_train, index_CG, alpha, time_limit, model, f) #model=0 -> no warm start else update the objective function
    is_L1_SVM = (len(index_CG) == N)
    

#---Infinite loop until all the variables have non reduced cost
    while True:
        aux += 1
        write_and_print('Round '+str(aux), f)
        
    #---Solve the problem to get the dual solution
        model.optimize()
        write_and_print('Time optimizing = '+str(time.time()-start), f)

    #---Compute all reduce cost and look for variable with negative reduced costs
        dual_slack       = [model.getConstrByName('slack_'+str(i)).Pi for i in range(N)]
        violated_columns = []
        
        
    #---THE FOLLOWING DOESNT HAPPEN FOR L1 SVM
        for column in set(range(P))-set(index_CG):
            reduced_cost = np.sum([y_train[i]*X_train[i,column]*dual_slack[i] for i in range(N)])
            reduced_cost = alpha  + min(reduced_cost, -reduced_cost)

            if reduced_cost < -epsilon_RC:
                violated_columns.append(column)
                
        
        
    #---Add the column with negative reduced costs
        n_columns_to_add = len(violated_columns)

        if n_columns_to_add>0:
            write_and_print('Number of columns added: '+str(n_columns_to_add), f)

            for i in range(n_columns_to_add):
                column_to_add = violated_columns[i]
                model = add_column_L1_SVM(X_train, y_train, model, column_to_add, range(N), alpha) 
                model.update()

                index_CG.append(column_to_add)

        
    #---WE ALWAYS BREAK FOR L1 SVM   
        else:
            break 



#---Solution
    beta_plus  = np.zeros(P)
    beta_minus = np.zeros(P)

    for i in index_CG:
        beta_plus[i]  = model.getVarByName('beta_+_'+str(i)).X 
        beta_minus[i] = model.getVarByName('beta_-_'+str(i)).X 

    beta    = np.round(np.array(beta_plus) - np.array(beta_minus),6)
    support = np.where(beta!=0)[0]
    write_and_print('\nObj value   = '+str(model.ObjVal), f)
    write_and_print('\nLen support = '+str(len(support)), f)


    solution_dual = np.array([model.getConstrByName('slack_'+str(index)).Pi for index in range(N)])
    support_dual  = np.where(solution_dual!=0)[0]
    write_and_print('Len support dual = '+str(len(support_dual)), f)


#---IF DELETE APPROACH, then remove non basic features
    
    if delete_columns and not is_L1_SVM: #CANNOT HAPPEN FOR L1 SVM

        idx_to_removes = list(set(index_CG) - set(support))

        betas_to_remove = np.array([model.getVarByName(name="beta_+_"+str(i)) for i in idx_to_removes] + 
                                   [model.getVarByName(name="beta_-_"+str(i)) for i in idx_to_removes])

        for idx_to_remove in idx_to_removes:
            index_CG.remove(idx_to_remove)

        for beta_to_remove in betas_to_remove:
            model.remove(beta_to_remove)
        model.update()


        
    
    time_CG = time.time()-start    


#---End
    write_and_print('Time = '+str(time_CG), f)
    return beta, support, time_CG, model, index_CG, model.ObjVal









