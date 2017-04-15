import numpy as np
import random
from scipy.stats import norm
import math

import sys
sys.path.append('../algorithms')
from heuristics_classification import *


def shuffle(X, y):

#X: array of size (N,P)
#y: list of size (N,)

    N, P = X.shape
    aux = np.concatenate([X,np.array(y).reshape(N,1)], axis=1)
    np.random.shuffle(aux)

    X = [aux[i,:P] for i in range(N)]
    y = [aux[i,P:] for i in range(N)]

    X = np.array(X).reshape(N,P)
    y = np.array(y).reshape(N,)

    return X,y








def simulate_data_classification(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f):



#RHO : correlation_coefficient
#TAU : controle entre mu + et - OR SNR
#SEED X : for random simulation

#SIGMA_1 = rho^(i-j), N(mu_+, Sigma), N(mu_-, Sigma)
#SIGMA_2 = rho^(i-j), sgn( N(0, Sigma)*mu_+ + epsilon)
#SIGMA_4 = rho,       N(mu_+, Sigma), N(mu_-, Sigma)
#SIGMA_5 = rho,       sgn( N(0, Sigma)*mu_+ + epsilon)



    np.random.seed(seed=seed_X)
    

#------------BETA-------------
    u_positive = np.zeros(P)

    if(type_Sigma==1):
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]  #equi-spaced k0
        u_positive[index] = tau_SNR*np.ones(k0)

    elif(type_Sigma==2):
        index = random.sample(xrange(P),k0)
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]
        u_positive[:k0] = np.cumsum((tau_SNR/k0)*np.ones(k0))

    elif(type_Sigma==3 or type_Sigma==5):
        u_positive[:k0] = np.ones(k0)

    elif(type_Sigma == 4):
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]  #equi-spaced k0
        u_positive[index] = np.ones(k0)
    
    
    u_negative = -u_positive


#------------SIGMA-------------
    

    if(type_Sigma==1 or type_Sigma==4 or type_Sigma==3):
        Sigma = np.zeros(shape=(P,P))

        for i in range(P):
            for j in range(P):
                Sigma[i,j]=rho**(abs(i-j))

    elif(type_Sigma==2 or type_Sigma==5):
        Sigma = rho*np.ones(shape=(P,P)) + (1-rho)*np.identity(P)




#------------X_train and X_test-------------
    X_train = np.zeros(shape=(N,P))
    y_train = []
    
    X_test=np.zeros(shape=(N,P))
    y_test = []
    
    
#---CASE 1 AND 4: RHO^(i-j)
    if(type_Sigma==1):

        L = np.linalg.cholesky(Sigma)

    #------------X_train-------------
        u_plus = np.random.normal(size=(P,N/2))
        X_plus = np.dot(L, u_plus).T + u_positive
        y_plus = np.ones(N/2)

        u_minus = np.random.normal(size=(P,N/2))
        X_minus = np.dot(L, u_minus).T + u_negative
        y_minus = -np.ones(N/2)

    #---Concatenate
        X_train = np.concatenate([X_plus, X_minus])
        y_train = np.concatenate([y_plus, y_minus])
        X_train, y_train = shuffle(X_train, y_train)


    #------------X_test-------------
        u_plus = np.random.normal(size=(P,N/2))
        X_plus = np.dot(L, u_plus).T + u_positive
        y_plus = np.ones(N/2)

        u_minus = np.random.normal(size=(P,N/2))
        X_minus = np.dot(L, u_minus).T + u_negative
        y_minus = -np.ones(N/2)

    #---Concatenate
        X_test = np.concatenate([X_plus, X_minus])
        y_test = np.concatenate([y_plus, y_minus])
        X_test, y_test = shuffle(X_test, y_test)






#---CASE 2: RHO and SGN(X MU + EPSILON)
    if(type_Sigma==2):


    #------------X_train-------------
        X0_plus = np.random.normal(u_positive, 1, size=(N/2,1))
        Xi_plus = np.random.normal(u_positive, 1, size=(N/2,P))
        
        for i in range(N/2):
            X_plus[i,:] = math.sqrt(rho)*X0_plus[i] + (1-rho)*Xi_plus[i,:]
        y_plus = np.ones(N/2)


        X0_minus = np.random.normal(u_negative, 1, size=(N/2,1))
        Xi_minus = np.random.normal(u_negative, 1, size=(N/2,P))
        
        for i in range(N/2):
            X_plus[i,:] = math.sqrt(rho)*X0_minus[i] + (1-rho)*Xi_minus[i,:]
        y_minus = np.ones(N/2)


    #---Concatenate
        X_train = np.concatenate([X_plus, X_minus])
        y_train = np.concatenate([y_plus, y_minus])
        X_train, y_train = shuffle(X_train, y_train)



    #------------X_test-------------
        X0_plus = np.random.normal(u_positive, 1, size=(N/2,1))
        Xi_plus = np.random.normal(u_positive, 1, size=(N/2,P))
        
        for i in range(N/2):
            X_plus[i,:] = math.sqrt(rho)*X0_plus[i] + (1-rho)*Xi_plus[i,:]
        y_plus = np.ones(N/2)


        X0_minus = np.random.normal(u_negative, 1, size=(N/2,1))
        Xi_minus = np.random.normal(u_negative, 1, size=(N/2,P))
        
        for i in range(N/2):
            X_plus[i,:] = math.sqrt(rho)*X0_minus[i] + (1-rho)*Xi_minus[i,:]
        y_minus = np.ones(N/2)


    #---Concatenate
        X_test = np.concatenate([X_plus, X_minus])
        y_test = np.concatenate([y_plus, y_minus])
        X_test, y_test = shuffle(X_test, y_test)







#---CASE 4: RHO^(i-j)
    elif(type_Sigma==4):

        std_eps = np.sqrt(np.dot(X_train, u_positive).var() / float(tau_SNR**2))
        L = np.linalg.cholesky(Sigma)

    #------------X_train-------------
        u = np.random.normal(size=(P,N))
        X_train = np.dot(L, u).T

        eps_train = np.random.normal(0, std_eps, N)
        y_train   = np.sign(np.dot(X_train, u_positive) + eps_train)


    #------------X_test-------------
        u = np.random.normal(size=(P,N))
        X_test = np.dot(L, u).T

        eps_test = np.random.normal(0, std_eps, N)
        y_test   = np.sign(np.dot(X_test, u_positive) + eps_test)








#---CASE 5: RHO and SGN(X MU + EPSILON)
    if(type_Sigma==5):

        std_eps = np.sqrt(np.dot(X_train, u_positive).var() / float(tau_SNR**2))

    #------------X_train-------------
        X0 = np.random.normal(size=(N,1))
        Xi = np.random.normal(size=(N,P))
        
        for i in range(N):
            X_train[i,:] = math.sqrt(rho)*X0[i] + (1-rho)*Xi[i,:]

        eps_train = np.random.normal(0, std_eps, N)
        y_train   = np.sign(np.dot(X_train, u_positive) + eps_train)

        #SHUFFLE
        X_train, y_train = shuffle(X_train, y_train)



    #------------X_test-------------
        X0 = np.random.normal(size=(N,1))
        Xi = np.random.normal(size=(N,P))
        
        for i in range(N):
            X_test[i,:] = math.sqrt(rho)*X0[i] + (1-rho)*Xi[i,:]

        eps_test = np.random.normal(0, std_eps, N)
        y_test   = np.sign(np.dot(X_test, u_positive) + eps_test)

        #SHUFFLE
        X_train, y_train = shuffle(X_train, y_train)





#---CASE 3
    if(type_Sigma==3):
    #------------X_train-------------
        for i in range(N):
            X_train[i,:] = np.random.multivariate_normal(np.zeros(P),Sigma,1)
            X_train_u_positive = np.dot(X_train, u_positive)

            probas = [norm.cdf(X_train_u_positive[i]) for i in range(N)]
            randoms = np.random.rand(P)
            y_train = [2*(randoms[i]<probas[i]).astype(int)-1 for i in range(N)]
            
            
    #------------X_test-------------
        for i in range(N):
            X_test[i,:] = np.random.multivariate_normal(zeros_P,Sigma,1)
            X_test_u_positive = np.dot(X_test, u_positive)
            
            probas = [norm.cdf(X_test_u_positive[i]) for i in range(N)]
            randoms = np.random.rand(P)
            y_test = [2*(randoms[i]<probas[i]).astype(int)-1 for i in range(N)]




#------------NORMALIZE------------- 
    
#---Normalize all the X columns
    l2_X_train = []

    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2)


    write_and_print('\nDATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+' Sigma='+str(type_Sigma)+' Seed_X='+str(seed_X), f)

    return X_train, X_test, l2_X_train, y_train, y_test, u_positive


