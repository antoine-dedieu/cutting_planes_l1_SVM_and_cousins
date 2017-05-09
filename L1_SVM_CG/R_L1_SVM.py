#R package
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

import time
import math
import numpy as np

import sys
sys.path.append('../real_datasets')
from process_data_real_datasets import *



def penalizedSVM_R_L1_SVM(X_train, y_train, alpha, f):

#---Create R objects
	N,P = X_train.shape
	penalizedSVM = importr("penalizedSVM")
	
	rpy2.robjects.numpy2ri.activate()
	nr,nc     = X_train.shape
	X_train_R = robjects.r.matrix(X_train, nrow=nr, ncol=nc)
	y_train_R = robjects.r.seq(y_train)


#---Train
	start = time.time()

	#Documentation found at https://cran.r-project.org/web/packages/penalizedSVM/penalizedSVM.pdf and code at http://research.cs.wisc.edu/dmi/svm/lpsvm/
	results = penalizedSVM.lpsvm(X_train, y_train, k=0, nu= 1./alpha) #definition of the problem
	time_R_L1 = time.time()-start

	coefs_R   = np.array(results[0])
	support_R = np.array(results[2])


	write_and_print('Len support = '+str(len(support_R)), f)
	write_and_print('Time = '+str(time_R_L1), f)


	beta_R = np.zeros(P)
	for i in range(len(support_R)):
		beta_R[int(support_R[i])-1] = coefs_R[i]


	return beta_R, support_R, time_R_L1





def SAM_R_L1_SVM(X_train, y_train, alpha, f):

#---Create R objects
	N,P = X_train.shape
	SAM = importr("SAM")

	rpy2.robjects.numpy2ri.activate()
	nr,nc     = X_train.shape
	X_train_R = robjects.r.matrix(X_train, nrow=nr, ncol=nc)
	y_train_R = robjects.r.seq(y_train)


#---Train
	start = time.time()

	#Documentation found at https://www.rdocumentation.org/packages/SAM/versions/1.0.2/topics/l1svm and http://proceedings.mlr.press/v22/zhao12/zhao12.pdf
	results = SAM.l1svm(X_train, y_train, math.log(P)/(100*N)) #definition of the problem
	time_R_L1 = time.time()-start

	#print results[3], math.log(P)/(100*N)
	
	beta_R = np.array(results[4])
	beta_R = beta_R.reshape(P,)

	support_R    = np.where(beta_R != 0)[0]
	write_and_print('Len support = '+str(len(support_R)), f)
	write_and_print('Time = '+str(time_R_L1), f)


	return beta_R, support_R, time_R_L1








def SVM_ADMM(X_train, y_train, alpha, f):

#---Create R objects
	N,P = X_train.shape
	svm = importr("svmadmm")

	rpy2.robjects.numpy2ri.activate()
	nr,nc     = X_train.shape
	X_train_R = robjects.r.matrix(X_train, nrow=nr, ncol=nc)
	y_train_R = robjects.r.seq(y_train)


#---Train
	start = time.time()

	#Documentation found at https://www.rdocumentation.org/packages/SAM/versions/1.0.2/topics/l1svm and http://proceedings.mlr.press/v22/zhao12/zhao12.pdf
	results = svm.svm_admm(X_train_R, y_train_R, type=2, kernel='linear')

	time_R_L1 = time.time()-start

	beta_R = np.array(results[0])
	b0     = float(beta_R[P])
	beta_R = beta_R[:P]

	support_R    = np.where(beta_R != 0)[0]
	write_and_print('Len support = '+str(len(support_R)), f)
	write_and_print('Time = '+str(time_R_L1), f)

	constraints = np.ones(N) - y_train*( np.dot(X_train[:, np.array(support_R)], beta_R) + b0*np.ones(N))
	obj_val     = np.sum([ max(constraints[i], 0) for i in range(N) ]) + alpha*np.sum(np.abs(beta_R))

	write_and_print('\nObj value   = '+str(obj_val), f)


	return beta_R, support_R, time_R_L1





