import numpy as np
import random
from scipy.stats import norm
import math

import sys
sys.path.append('../algorithms')
from heuristics_classification import *



def simulate_train_test_classification_sparsity(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f):


    X_train, y_train, u_positive = simulate_data_classification_sparsity(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)
    X_test,  y_test,  u_positive = simulate_data_classification_sparsity(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f)

    
#---Normalize all the X_train columns
    l2_X_train = []

    for i in range(P):
        l2 = np.linalg.norm(X_train[:,i])
        #l2 = np.std(X_train[:,i])
        l2_X_train.append(l2)        
        X_train[:,i] = X_train[:,i]/float(l2)


    write_and_print('\nDATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+' Sigma='+str(type_Sigma)+' Seed_X='+str(seed_X), f)

    return X_train, X_test, l2_X_train, y_train, y_test, u_positive







def simulate_data_classification_sparsity(type_Sigma, N, P, k0, rho, tau_SNR, seed_X, f):


#RHO : correlation_coefficient
#TAU : controle entre mu + et - OR SNR
#SEED X : for random simulation

#SIGMA_1 = rho^(i-j), N(mu_+, Sigma), N(mu_-, Sigma)     LDA
#SIGMA_2 = rho,       N(mu_+, Sigma), N(mu_-, Sigma)     LDA
#SIGMA_4 = rho^(i-j), sgn( N(0, Sigma)*mu_+ + epsilon)  
#SIGMA_5 = rho,       sgn( N(0, Sigma)*mu_+ + epsilon)



    np.random.seed(seed=seed_X)
    

#------------BETA-------------
    u_positive = np.zeros(P)

    if(type_Sigma==1):
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]  #equi-spaced k0
        u_positive[index] = tau_SNR*np.ones(k0)

    elif(type_Sigma==2):
        u_positive[:k0] = tau_SNR*np.ones(k0)


    elif(type_Sigma==3 or type_Sigma==5):
        u_positive[:k0] = np.ones(k0)

    elif(type_Sigma == 4):
        index = [(2*i+1)*P/(2*k0) for i in range(k0)]  #equi-spaced k0
        u_positive[index] = np.ones(k0)
    
    
    u_negative = -u_positive


#------------SIGMA-------------
    
#### LDA MODEL
    
    if type_Sigma == 1:
        Sigma = np.diag(np.ones(P))

        for i in range(k0):
            for j in range(k0):
                Sigma[i,j]=rho**(abs(i-j))



    if(type_Sigma==3 or type_Sigma==4):
        Sigma = np.zeros((P,P))

        for i in range(P):
            for j in range(P):
                Sigma[i,j]=rho**(abs(i-j))


## DO NOT NEED IF TYPE = 2, 5

    
#---CASE 1: RHO^(i-j)
    if type_Sigma == 1:

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





#---CASE 2: RHO 
    elif type_Sigma == 2:

        X_plus  = np.zeros((N/2, k0))
        X_minus = np.zeros((N/2, k0))


        X0_plus    = np.random.normal(size=(N/2,1))
        Xi_plus_k0 = np.random.normal(loc= 1./math.sqrt(1-rho)*u_positive[:k0], size=(N/2,k0))

        for i in range(N/2):
            X_plus[i,:k0] = math.sqrt(rho)*X0_plus[i] + math.sqrt(1-rho)*Xi_plus_k0[i,:]

        X_plus = np.concatenate([X_plus, np.random.normal(size=(N/2,P-k0))], axis=1)
        y_plus = np.ones(N/2)



        X0_minus = np.random.normal(size=(N/2,1))
        Xi_minus_k0 = np.random.normal(loc= 1./math.sqrt(1-rho)*u_negative[:k0], size=(N/2,k0))

        for i in range(N/2):
            X_minus[i,:k0] = math.sqrt(rho)*X0_minus[i] + math.sqrt(1-rho)*Xi_minus_k0[i,:]

        X_minus = np.concatenate([X_minus, np.random.normal(size=(N/2,P-k0))], axis=1)
        y_minus = - np.ones(N/2)


    #---Concatenate
        X_train = np.concatenate([X_plus, X_minus])
        y_train = np.concatenate([y_plus, y_minus])
        X_train, y_train = shuffle(X_train, y_train)





#---CASE 4: RHO^(i-j)
    elif type_Sigma == 4:

        L       = np.linalg.cholesky(Sigma)
        u       = np.random.normal(size=(P,N))
        X_train = np.dot(L, u).T

        std_eps   = np.sqrt(np.dot(X_train, u_positive).var() / float(tau_SNR**2))
        eps_train = np.random.normal(0, std_eps, N)
        y_train   = np.sign(np.dot(X_train, u_positive) + eps_train)






#---CASE 5: RHO and SGN(X MU + EPSILON)
    if type_Sigma == 5:

        X0 = np.random.normal(size=(N,1))
        Xi = np.random.normal(size=(N,P))
        
        for i in range(N):
            X_train[i,:] = math.sqrt(rho)*X0[i] + math.sqrt(1-rho)*Xi[i,:]

        std_eps   = np.sqrt(np.dot(X_train, u_positive).var() / float(tau_SNR**2))
        eps_train = np.random.normal(0, std_eps, N)
        y_train   = np.sign(np.dot(X_train, u_positive) + eps_train)





#---CASE 3
    if(type_Sigma==3):

        for i in range(N):
            X_train[i,:] = np.random.multivariate_normal(np.zeros(P),Sigma,1)
            X_train_u_positive = np.dot(X_train, u_positive)

            probas = [norm.cdf(X_train_u_positive[i]) for i in range(N)]
            randoms = np.random.rand(P)
            y_train = [2*(randoms[i]<probas[i]).astype(int)-1 for i in range(N)]
            
        

    return X_train, y_train, u_positive





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
