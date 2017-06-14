import os
import numpy as np
import sys

sys.path.append('../synthetic_datasets')
from simulate_data_classification import *






def store_ADMM_SVM_comparison(X, y, real_synthetic, train_test):


	current_path = os.path.dirname(os.path.realpath(__file__))
	data_train   = open(current_path+'/../../struct_svm_admm/data/'+str(real_synthetic)+'_dataset/data_'+str(train_test), 'w')
	N,P = X.shape

	for j in range(P):    
		X[:,j] *= l2_X[j]

	for i in range(N):
	    line  = str(int(1.5+0.5*y[i]))+' '
	    for j in range(P):
	        line += str(j+1)+':'+str(X[i,j])+' '
	    line += '\n'
	    data_train.write(line)


def check_ADMM_SVM_comparison(current_path):

	time          = open(current_path+'/time_EN')
	test_accuracy = open(current_path+'/accuracy_EN')

	for t in time:
		time = float(t)

	for t in test_accuracy:
		test_accuracy = float(t)

	return test_accuracy, time




def store_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, seed_X, alpha, single_double):


	current_path  = os.path.dirname(os.path.realpath(__file__))
	#current_path += '/../../LPsparse/data/synthetic_dataset/data_train_'+'N_'+str(N)+'_P_'+str(P)+'_seed_'+str(seed_X)+'/'
	current_path += '/../../LPsparse/data/synthetic_dataset/data_train_'+single_double


	A   = open(current_path+'/A',   'w')
	Aeq = open(current_path+'/Aeq', 'w')
	b   = open(current_path+'/b',   'w')
	beq = open(current_path+'/beq', 'w')
	c   = open(current_path+'/c',   'w')



#-------A
	A.write(str(N)+' '+str(N + 2*P + 2)+' '+str(0)+'\n')

	for i in range(N):
		vect = np.concatenate([-np.ones(N), -y_train[i]*X_train[i,:], y_train[i]*X_train[i,:], -np.array([y_train[i]]), np.array([y_train[i]]) ], axis=0)
		for j in range(N + 2*P + 2):
		    A.write(str(i+1)+' '+str(j+1)+' '+str(vect[j])+'\n')

	Aeq.write(str(0)+' '+str(N + 2*P + 2)+' '+str(0)+'\n')
	print vect


#-------b
	for i in range(N):
		b.write(str(-1.0)+'\n')

#-------c
	for i in range(N):
		c.write(str(1.0)+'\n')
	for i in range(2*P):
		c.write(str(alpha)+'\n')
	for i in range(2):
		c.write(str(0.0)+'\n')


#---META
	meta = open(current_path+'/meta', 'w')
	meta.write('nb '+str(N + 2*P + 2)+'\nnf 0 \nmI '+str(N)+'\nmE 0')








def check_AL_CD_comparison(X_train, y_train, l2_X_train, N, P, alpha, current_path, f):

	sol  = open(current_path+'/sol', 'r')
	time = open(current_path+'/time', 'r')

	for t in time:
		time = float(t)


	#for j in range(P):    
	#	X_train[:,j] *= l2_X_train[j]


	supp_AL_CD       = []
	beta_AL_CD_plus  = np.zeros(P)
	beta_AL_CD_minus = np.zeros(P)

	xi = np.zeros(N)

	b0_AL_CD_plus   = 0
	b0_AL_CD_minus  = 0


	for line in sol:
		line = line.split("\t")
		idx  = int(line[0])

		#supp_AL_CD.append(idx-1)
		if idx < N+1:
			xi[idx-1] = float(line[1].split("\n")[0] )
		elif idx < N+P+1:
			beta_AL_CD_plus[idx-N-1]  = float(line[1].split("\n")[0] )
			#beta_AL_CD_plus.append(float(line[1].split("\n")[0] ))
		elif idx < N+2*P+1:
			beta_AL_CD_minus[idx-N-P-1] = float(line[1].split("\n")[0] )
			#beta_AL_CD_minus.append(float(line[1].split("\n")[0] ))
		elif idx == N+2*P+1:
			b0_AL_CD_plus = float(line[1].split("\n")[0] )
		elif idx == N+2*P+1:
			b0_AL_CD_minus = float(line[1].split("\n")[0] )


	beta_AL_CD = (beta_AL_CD_plus - beta_AL_CD_minus)#*l2_X_train
	b0_AL_CD   = b0_AL_CD_plus   - b0_AL_CD_minus

	#constraints = np.ones(N) - y_train*( np.dot(X_train[:, supp_AL_CD], beta_AL_CD) + b0_AL_CD*np.ones(N))
	constraints = np.ones(N) - y_train*( np.dot(X_train, beta_AL_CD) + b0_AL_CD*np.ones(N))
	obj_val     = np.sum([max(constraints[i], 0) for i in range(N)]) + alpha*np.sum(np.abs(beta_AL_CD))

	write_and_print('\n\nTIME AL_CD   = '+str(time), f)   
	write_and_print('OBJ VAL AL_CD     = '+str(obj_val), f)   

	obj_val_bis  = np.sum(xi) + alpha*np.sum(np.abs(beta_AL_CD))
	write_and_print('OBJ VAL BIS AL_CD = '+str(obj_val_bis), f)   

	support = np.where(beta_AL_CD!=0)[0]
	write_and_print('Support = '+str(len(support)), f)   

	return obj_val_bis, time
