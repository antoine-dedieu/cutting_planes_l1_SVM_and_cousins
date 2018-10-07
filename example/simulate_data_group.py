import numpy as np
import random
from scipy.stats import norm
import math

from simulate_data import *




def simulate_data_group(N, P, group_to_feat, k0, rho, mu, seed_X):

####### INPUT #######

# N,P           : size of X 
# group_to_feat : set of features associated to each group
# k0            : number of relevant groups 
# rho           : coefficient of correlation for the covariance matrix Sigma
# mu            : used to define mu_+ and mu_-
# seed_X        : for random simulation


####### OUTPUT #######

# X, y: on each group, X is generated as x_+ \sim N(u_+, Sigma),  x_- \sim N(u_-, Sigma),  mu_+ =(mu,mu...mu, 0,...0), mu_- = -mu_+, Sigma_ij = rhos
# l2_X: l2_X contains the l2 norm of the columns of X (which have been normalized)


    np.random.seed(seed=seed_X)
    
### Define mu_positive
    mu_positive = np.zeros(P)
    for idx in range(k0): mu_positive[group_to_feat[idx]] = mu*np.ones(len(group_to_feat[idx]))

    mu_negative = -mu_positive



### Define X
    X_plus  = np.zeros((N/2, 0))
    X_minus = np.zeros((N/2, 0))

    G = len(group_to_feat)
    P = np.sum([len(group_to_feat[idx]) for idx in range(G)])


### First half
    X0_plus  = np.random.normal(size=(N/2, G))
    Xi_plus  = np.random.normal(loc=  1./float(math.sqrt(1-rho))*mu_positive, size=(N/2, P))

    #Simulate group by group
    for idx in range(G): 
        sub_X0_plus = np.array([X0_plus[:, idx] for _ in group_to_feat[idx]]).T
        sub_Xi_plus = Xi_plus[:, group_to_feat[idx]]
        X_plus      = np.concatenate([X_plus, math.sqrt(rho)*sub_X0_plus + math.sqrt(1-rho)*sub_Xi_plus ], axis=1)


### Second half
    X0_minus = np.random.normal(size=(N/2, G))
    Xi_minus = np.random.normal(loc=  1./float(math.sqrt(1-rho))*mu_negative, size=(N/2, P))

    #Simulate group by group
    for idx in range(G): 
        sub_X0_minus = np.array([X0_minus[:, idx] for _ in group_to_feat[idx]]).T
        sub_Xi_minus = Xi_minus[:, group_to_feat[idx]]
        X_minus      = np.concatenate([X_minus, math.sqrt(rho)*sub_X0_minus + math.sqrt(1-rho)*sub_Xi_minus ], axis=1)


### Concatenate and shuffle
    X = np.concatenate([X_plus, X_minus])
    y = np.concatenate([np.ones(N/2), -np.ones(N/2)])
    X, y = shuffle(X, y)


### Normalize X    
    l2_X = []
    for i in range(P):
        l2 = np.linalg.norm(X[:,i])
        l2_X.append(l2)        
        X[:,i] = X[:,i]/float(l2)


    print 'DATA CREATED for N='+str(N)+', P='+str(P)+', k0='+str(k0)+' Rho='+str(rho)+' Seed_X='+str(seed_X)
    return X, l2_X, y


