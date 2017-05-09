import numpy as np
from sklearn import svm

from Gurobi_SVM import *




def SVM_support(type_loss, type_penalization, X, y, alpha, beta_end, time_limit):

#TYPE_LOSS = 1 : HINGE LOSS 
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 
#TYPE_PENALIZATION = 2 : L2


    NN,PP = len(X),len(X[0])
    
    beta_non_zeros = np.where(beta_end!=0)[0]
    r = len(beta_non_zeros)

    
#---Support not empty
    if(r>0):
      
    #---Restrict the space
        X_support = np.zeros(shape=(NN,r))
        for i in range(r):
            X_support[:,i] = X[:,beta_non_zeros[i]]


    #---ALGORITHM
        if type_loss=='hinge' and type_penalization=='l1' :
            
        #---Gurobi without model or warm-start
            beta_support_non_zeros, beta_0, _, _, _ = Gurobi_SVM('hinge', 'l1', 'no_l0', X_support, y, 0, alpha, time_limit, 0, 0)


        else:
        #---Run SVM for primal and correct problem
            dict_dual = {'l1':False, 'l2':True}
            
            estimator = svm.LinearSVC(penalty=type_penalization, loss= type_loss, dual=dict_dual[type_penalization], C=1./(2*alpha))
            estimator.fit(X_support, y)

            beta_support_non_zeros = estimator.coef_[0]
            beta_0  = estimator.intercept_[0] 
        


    #---Full support solution
        beta_SVM_support = np.zeros(PP)
        for i in range(r):
            beta_SVM_support[beta_non_zeros[i]] = beta_support_non_zeros[i]

            
#---Support empty
    else:
        beta_SVM_support = np.zeros(PP)
        beta_0 = 0
       
    
#---Compute error
    dot_product_y = y*(np.dot(X,beta_SVM_support)+ beta_0*np.ones(NN))

    dict_loss = {'hinge': np.sum([max(0, 1-dot_product_y[i])  for i in range(NN)]),
                 'squared_hinge': np.sum([max(0, 1-dot_product_y[i])**2  for i in range(NN)]) }
    
    dict_penalization = {'l1': np.sum(np.abs(beta_SVM_support)),
                         'l2': np.linalg.norm(beta_SVM_support)**2}

    error = dict_loss[type_loss] + alpha*dict_penalization[type_penalization]
        


    return beta_SVM_support, beta_0, error







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







def algorithm1_classification(type_loss, type_penalization, X, y, K0, alpha, tau, start, X_add, L, epsilon, time_limit):
    
#TYPE_LOSS = 1 : HINGE LOSS -> function of the parameter tau
#TYPE_LOSS = 2 : SQUARED HINGE LOSS

#TYPE_PENALIZATION = 1 : L1 -> soft thresholding
#TYPE_PENALIZATION = 2 : L2


#CAREFULL : X can be a submatrix of the original one
#START : used when K0/alpha increases/decreases to warm start -> OF SIZE PP+1
#EPSILON : stop the convergence


    
#BETA_M : the last component is beta_0 the origin coefficient

    X_add = np.ones((N,P+1))
    X_add[:,:P] = X_train

    L = np.linalg.norm(np.dot(X_add.T,X_add))
    
    NN, PP = X.shape
    
    #Initialisation
    old_beta = -np.ones(PP+1)
        
    if(len(start)==0):
        beta_m=np.zeros(PP+1)
    else:
        beta_m=start

    
#---MAIN LOOP   
    test =0
    while(not np.array_equal(old_support, support) and test <50):
    #while(np.linalg.norm(beta_m-old_beta)>epsilon and test <200): 
        #print np.where(beta_m!=0), np.linalg.norm(beta_m-old_beta)
        test+=1

        #Penalization only on first PP element
        beta_m_no_b0 = beta_m
        beta_m_no_b0[PP] = 0

        
        aux = np.ones(NN)- y*np.dot(X_add,beta_m)

        
    #---LOSS
    #---Hinge loss
        if (type_loss=='hinge'):
            w_tau = [min(1, abs(aux[i])/tau)*np.sign(aux[i])  for i in range(NN)]
            aux = [-0.5*y[i]*(1+w_tau[i])*X_add[i,:] for i in range(NN)]

            Lipchtiz_coeff = L/(4*tau) 


    #---Squared hinge loss
        elif (type_loss=='squared_hinge'):
            xi = [max(0,aux[i]) for i in range(NN)]
            aux = [-2*y[i]*xi[i]*X_add[i,:] for i in range(NN)]

            Lipchtiz_coeff = 2*L
        

    #---Gradient descent
        old_beta = beta_m 
        eta_m = beta_m - 1./Lipchtiz_coeff*(np.sum(aux,axis=0))
        
    
    #---THRESHOLDING
        beta_m = np.zeros(PP+1)
        beta_m[PP] = eta_m[PP]
        index = np.abs(eta_m[:PP]).argsort()[::-1][:K0]
        
        dict_thresholding = {'l1': soft_thresholding_l1,
                             'l2': soft_thresholding_l2}
        beta_m[index] = np.array([ dict_thresholding[type_penalization](eta_m[i],2.*alpha/L) for i in index])
        
    
    
    #print w_tau, test
    print 'Number of iterations: ' +str(test)
    
                                   
#---WHEN THE LOOP IS OVER, WE RUN THE EXACT SOLUTION ON ITS SUPPORT

    if (type_loss=='squared_hinge'):
        #Only smoothin on support for squared hinge loss -> hinge loss only last step
        
        dot_product_y = y*(np.dot(X,beta_m[:PP])+ beta_m[PP]*np.ones(NN))
        error = np.sum([max(0, 1-dot_product_y[i])**2  for i in range(NN)]) 
        dict_error_penalization = {'l1' : alpha*np.sum(np.abs(beta_m[:PP])),
                                   'l2' : alpha*np.linalg.norm(beta_m[:PP])}
        error += dict_error_penalization[type_penalization]
        print error#, dict_error_penalization[type_penalization]
        
        beta_SVM_support, beta_0, train_error = SVM_support('squared_hinge', type_penalization, X, y, alpha, beta_m[:PP], time_limit)
        
        
    elif (type_loss=='hinge'):
        #Only smoothin on support for squared hinge loss -> hinge loss only last step
        beta_SVM_support, beta_0, train_error = beta_m[:PP], beta_m[PP], 0
        
        
    return beta_SVM_support, beta_0, train_error







def algorithm1_smoothing_hinge_loss(type_penalization, X, y, K0, alpha, start, X_add, L, epsilon, time_limit):
    
#Apply the smoothing technique from the best subset selection
    
    NN, PP = X.shape
    old_beta = -np.ones(PP)
    beta = np.zeros(PP)
    beta_start = []
    
    tau = 0.5
    
    idx = -1
    while(np.linalg.norm(beta-old_beta)>epsilon and idx<20):
        idx +=1
        old_beta = beta
        
        beta, beta0, _ = algorithm1_classification('hinge', type_penalization, X, y, K0, alpha, tau, beta_start, X_add, L, epsilon, time_limit)
        
        dot_product_y = y*(np.dot(X,beta)+ beta0*np.ones(NN))
        error = np.sum([max(0, 1-dot_product_y[i])  for i in range(NN)]) + alpha*np.linalg.norm(beta)**2
        print error #, beta[np.where(beta!=0)[0]], beta0
        
        beta_start = beta0*np.ones(PP+1)
        beta_start[:PP] = beta
        tau = 0.8*tau
        
#---Last step: solving on support
    beta_SVM_support, beta_0, train_error = SVM_support('hinge', type_penalization, X, y, alpha, beta, time_limit)
    
    
    return beta_SVM_support, beta_0, train_error





def algorithm1_unified(type_loss, type_penalization, X, y, K0, alpha, start, X_add, L, epsilon, time_limit):

    if type_loss == 'hinge':
        return algorithm1_smoothing_hinge_loss(type_penalization, X, y, K0, alpha, start, X_add, L, epsilon, time_limit)
    elif type_loss == 'squared_hinge':
        return algorithm1_classification(type_loss, type_penalization, X, y, K0, alpha, 0, start, X_add, L, epsilon, time_limit) # tau is 0



