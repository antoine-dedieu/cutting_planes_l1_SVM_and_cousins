import numpy as np

def power_method(X):
#Computes the highest eigenvalue of XTX
    
    P = X.shape[1]

    highest_eigvctr     = np.random.rand(P)
    old_highest_eigvctr = -1
    
    while(np.linalg.norm(highest_eigvctr - old_highest_eigvctr)>1e-2):
        old_highest_eigvctr = highest_eigvctr
        highest_eigvctr     = np.dot(X.T, np.dot(X, highest_eigvctr))
        highest_eigvctr    /= np.linalg.norm(highest_eigvctr)
    
    X_highest_eig = np.dot(X, highest_eigvctr)
    highest_eig   = np.dot(X_highest_eig.T, X_highest_eig)/np.linalg.norm(highest_eigvctr)
    
    return highest_eig