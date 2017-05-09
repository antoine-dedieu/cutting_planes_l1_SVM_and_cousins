import numpy as np
import time
import math

import sys

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


# STARTING POINT FOR L1-SVM

def smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, beta_start, X_add, highest_eig, tau, n_iter, f):
    
#TYPE_PENALIZATION = 1 : L1 -> soft thresholding
#TYPE_PENALIZATION = 2 : L2
#BETA_M : the last component is beta_0 the origin coefficient
    
    
#---Initialisation
    start_time = time.time()
    N, P  = X.shape

    old_beta = -np.ones(P+1)



    beta_m = beta_start

    

#---MAIN LOOP   
    test  =0
    t_AGD_old =1
    t_AGD     =1
    eta_m_old = beta_start
    ones_N    = np.ones(N)


    if (type_loss=='hinge'):
        Lipchtiz_coeff  = highest_eig/(4*tau) 
    elif (type_loss=='squared_hinge'):
        Lipchtiz_coeff = 2*highest_eig
    


    while(np.linalg.norm(beta_m-old_beta)>1e-4 and test < n_iter): 
    #while(not np.array_equal(old_support, support) and test <20):
        
        #print np.linalg.norm(beta_m-old_beta)
        test+=1
        aux = ones_N - y*np.dot(X_add,beta_m)
        

    #---Hinge loss
        if (type_loss=='hinge'):
            w_tau           = [min(1, abs(aux[i])/(2*tau))*np.sign(aux[i])  for i in range(N)]
            #gradient_loss   = -0.5*np.sum([y[i]*(1+w_tau[i])*X_add[i,:] for i in range(N)], axis=0)
            gradient_aux  = np.array([y[i]*(1+w_tau[i]) for i in range(N)])
            gradient_loss = -0.5*np.dot(X_add.T, gradient_aux)



        if (type_loss=='squared_hinge'):
            xi            = [max(0,aux[i]) for i in range(N)]
            gradient_loss = -2*np.sum([y[i]*xi[i]*X_add[i,:] for i in range(N)], axis=0)



    #---Gradient descent
        old_beta    = beta_m 
        
        grad = beta_m - 1/float(Lipchtiz_coeff)*gradient_loss


    #---Thresholding of top 100 guys !
        #eta_m    = np.zeros(P+1)
        #eta_m[P] = grad[P]
        #index    = np.abs(eta_m[:P]).argsort()[::-1][:50]

        dict_thresholding = {'l1': soft_thresholding_l1,
                             'l2': soft_thresholding_l2}
        eta_m = np.array([ dict_thresholding[type_penalization](grad[i], 2.*alpha/Lipchtiz_coeff) for i in range(P)] + [grad[P]])
        
        

    #---AGD
        t_AGD     = (1 + math.sqrt(1+4*t_AGD_old**2))/2.
        aux_t_AGD = (t_AGD_old-1)/t_AGD

        beta_m     = eta_m + aux_t_AGD * (eta_m - eta_m_old)

        t_AGD_old = t_AGD
        eta_m_old = eta_m



        #support = np.where(beta_m != 0)[0]

        #print'\n Iteration: '+str(test)
        #print np.max(np.abs(beta_m[support]))

    
    

    write_and_print('\nNumber of iterations: ' +str(test), f)
    #write_and_print('\nNumber of iterations: ' +str(test), f)

#---Keep top 50
    #index    = np.abs(beta_m[:P]).argsort()[::-1][:50]
    #index = range(P)
    b0 = beta_m[P]/math.sqrt(N)
    #write_and_print('intercept: '+str(b0), f) #very small


#---Support
    idx_columns_smoothing   = np.where(beta_m[:P] !=0)[0]
    write_and_print('Len support smoothing: '+str(idx_columns_smoothing.shape[0]), f)


#---Constraints
    #constraints = ones_N - y*( np.dot(X[:,idx_columns_smoothing], beta_m[idx_columns_smoothing]) + b0*ones_N)
    constraints = 1.25*ones_N - y*( np.dot(X[:,idx_columns_smoothing], beta_m[idx_columns_smoothing]))
    idx_samples_smoothing = np.arange(N)[constraints >= 0]
    write_and_print('Len dual smoothing: '+str(idx_samples_smoothing.shape[0]), f)
    write_and_print('Convergence rate    : ' +str(round(np.linalg.norm(beta_m-old_beta), 3)), f) 
    
    time_smoothing = time.time()-start_time
    write_and_print('Time smoothing: '+str(round(time_smoothing,3)), f)


    return idx_samples_smoothing.tolist(), idx_columns_smoothing.tolist(), time_smoothing, beta_m



    








def loop_smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, tau_max, n_loop, time_limit, n_iter, f):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection
    
    start_time = time.time()
    N, P = X.shape
    old_beta = -np.ones(P+1)

#---New matrix and SVD
    X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
    X_add[:,:P] = X

    highest_eig     = power_method(X_add, P+1)


    beta_smoothing  = np.zeros(P+1)
    time_smoothing_sum = 0

    
    tau = tau_max
    
    test = 0
    while(np.linalg.norm(beta_smoothing-old_beta)>1e-4 and test < n_loop): 

        test += 1
        old_beta = beta_smoothing
        
        idx_samples, idx_columns, time_smoothing, beta_smoothing = smoothing_hinge_loss(type_loss, type_penalization, X, y, alpha, beta_smoothing, X_add, highest_eig, tau, n_iter, f)

    #---Update parameters
        time_smoothing_sum += time_smoothing
        tau = 0.7*tau


    #print beta_smoothing[idx_columns]

    time_smoothing_tot = time.time()-start_time
    write_and_print('\nNumber of iterations                       : '+str(test), f)
    write_and_print('Total time smoothing for '+str(type_loss)+': '+str(round(time_smoothing_tot, 3)), f)

    return idx_samples, idx_columns, time_smoothing_sum, beta_smoothing[:P]






def loop_smoothing_hinge_loss_restriction(type_loss, type_penalization, X, y, alpha, n_loop, time_limit, f):
    
#n_loop: how many times should we run the loop ?
#Apply the smoothing technique from the best subset selection

    N, P = X.shape
    #old_beta = -np.ones(P+1)

#---New matrix and SVD
    X_add       = 1/math.sqrt(N)*np.ones((N, P+1))
    X_add[:,:P] = X
    U, diag, V  = np.linalg.svd(X_add)
    max_svd     = np.max(np.abs(diag))


    beta_smoothing_restricted  = np.zeros(P+1)
    X_reduced   = X
    idx_columns = np.arange(P)

    time_smoothing_sum = 0

    
    tau = 1
    
    for i in range(n_loop):
        old_beta = beta_smoothing_restricted
        
        _, idx_columns_restricted, time_smoothing, aux_beta_smoothing_restricted = smoothing_hinge_loss(type_loss, type_penalization, X_reduced, y, alpha, beta_smoothing_restricted, X_add, max_svd, tau, f)
     

    #---Restrict to columns
        X_reduced    = np.array([X_reduced[:,j] for j in idx_columns_restricted]).T

        P_reduced   = len(idx_columns_restricted)

        X_add               = 1/math.sqrt(N)*np.ones((N, P_reduced+1))
        X_add[:,:P_reduced] = X_reduced

        U, diag, V  = np.linalg.svd(X_add)
        max_svd     = np.max(np.abs(diag))

        beta_smoothing_restricted             = np.zeros(P_reduced+1)
        beta_smoothing_restricted[:P_reduced] = aux_beta_smoothing_restricted[idx_columns_restricted]


    #---Update parameters
        idx_columns  = idx_columns[idx_columns_restricted]
        time_smoothing_sum += time_smoothing
        tau = 0.7 *tau


#---Final estimator
    beta_smoothing = np.zeros(P+1)
    for i in range(len(idx_columns)):
        beta_smoothing[idx_columns[i]] = beta_smoothing_restricted[i]

    write_and_print('Time smoothing for '+str(type_loss)+': '+str(time_smoothing_sum), f)

    #if len(support_smoothing) >500:
    #    beta = beta_smoothing
    #    support_smoothing     = np.abs(beta[:P]).argsort()[::-1][:50]
    #    beta_smoothing        = np.zeros(P)
    #    beta_smoothing[support_smoothing] = beta[support_smoothing]
    #    support_smoothing = support_smoothing.tolist()
        
    return idx_columns.tolist(), idx_columns.tolist(), time_smoothing_sum, beta_smoothing
        



#POWER METHOD to compute the SVD of XTX

def power_method(X, P):
    
    highest_eigvctr     = np.random.rand(P)
    old_highest_eigvctr = -1
    
    while(np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2):
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))
        highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
    
    X_highest_eig = np.dot(X, highest_eigvctr)
    highest_eig   = np.dot(X_highest_eig.T, X_highest_eig)/np.linalg.norm(highest_eigvctr)
    return highest_eig



def soft_thresholding_l1(c,alpha):
    if(alpha>=abs(c)):
        return 0
    else:
        if (c>=0):
            return c-alpha
        else:
            return c+alpha
    
    
def soft_thresholding_l2(c,alpha):
    return c/float(1+2*alpha)




