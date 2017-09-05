import numpy as np
import os

import sys
sys.path.append('../synthetic_datasets')
from simulate_data_classification import *


def write_and_print(text,f):
    print text
    f.write('\n'+text)




def process_real_datasets(name_real_dataset):

	if name_real_dataset == 'ovarian'
		g=open('../../../datasets/datasets_unprocessed/ovarian.data',"r")
		dict_type = {'Normal\n':-1,'Cancer\n':1}


	elif name_real_dataset == 'lungCancer'
		g=open('../../../datasets/datasets_unprocessed/lungCancer_train.data',"r")
		h=open('../../../datasets/datasets_unprocessed/lungCancer_test.data',"r")
		dict_type = {'Mesothelioma\r\n':-1,'ADCA\r\n':1}


	elif name_real_dataset == 'leukemia'
		g=open('../../../datasets/datasets_unprocessed/leukemia_train.data',"r")
		h=open('../../../datasets/datasets_unprocessed/leukemia_test.data',"r")
		dict_type = {'ALL\r\r\n':-1,'AML\r\r\n':1}



	elif name_real_dataset == 'radsens'
		N_real, P_real = 58, 12625
		train_data   = open('../../../datasets/datasets_unprocessed/radsens.x.txt',"r")
		train_labels = open('../../../datasets/datasets_unprocessed/radsens.y.txt',"r")


	elif name_real_dataset == '14cancer'
		N_real, P_real = 198, 16063
		train_data = open('../../../datasets/datasets_unprocessed/14cancer.xtrain.txt',"r")
		valid_data = open('../../../datasets/datasets_unprocessed/14cancer.xtest.txt',"r")

		train_labels = open('../../../datasets/datasets_unprocessed/14cancer.ytrain.txt',"r")
		valid_labels = open('../../../datasets/datasets_unprocessed/14cancer.ytest.txt',"r")


	elif name_real_dataset == 'dexter'
		N_real, P_real = 600, 20000
		train_data = open('../../datasets/dexter_train.data.txt',"r")
		valid_data = open('../../datasets/dexter_valid.data.txt',"r")

		train_labels   = open('../../datasets/dexter_train.labels.txt',"r")
		valid_labels = open('../../datasets/dexter_valid.labels.txt',"r")


	elif name_real_dataset == 'gisette'
		N_real, P_real = 7000, 5000
		train_data = open('../../datasets/gisette_train.data.txt',"r")
		valid_data = open('../../datasets/gisette_valid.data.txt',"r")

		train_labels = open('../../datasets/gisette_train.labels.txt',"r")
		valid_labels = open('../../datasets/gisette_valid.labels.txt',"r")


	elif name_real_dataset == 'madelon'
		N_real, P_real = 7000, 5000
		train_data = open('../../datasets/madelon_train.data.txt',"r")
		valid_data = open('../../datasets/madelon_valid.data.txt',"r")

		train_labels = open('../../datasets/madelon_train.labels.txt',"r")
		valid_labels = open('../../datasets/madelon_valid.labels.txt',"r")


	elif name_real_dataset == 'arcene'
		N_real, P_real = 200, 100000
		train_data = open('../../datasets/arcene_train.data.txt',"r")
		valid_data = open('../../datasets/arcene_valid.data.txt',"r")

		train_labels = open('../../datasets/arcene_train.labels.txt',"r")
		valid_labels = open('../../datasets/arcene_valid.labels.txt',"r")

	elif name_real_dataset == 'farm-ads'
		N_real, P_real = 4143, 54878
		data = open('../../../datasets/datasets_unprocessed/farm-ads-vect',"r")



#---------------------------------------
    X0, y = pd.DataFrame(), []


    if name_real_dataset in ['leukemia', 'lungCancer']
        for i in [g,h]:
            for line in i:
                line,data_line=line.split(",")[::-1],[]
                y.append(dict_type[str(line[0])])

                for aux in line[1:len(line)]:
                    data_line.append(float(aux))
                X0=pd.concat([X0, pd.DataFrame(data_line).T])


    if name_real_dataset == 
        X = np.zeros((N_real, P_real))
        y = np.zeros(N_real)
        aux = -1

        for line in data:
            aux += 1
            line = line.split(" ")
            y[aux] = line[0]

            for j in range(1, len(line)):
                couple = line[j].split(":")
                
                if couple[1][-2:]!='\n': #if line not ended
                    X[aux, int(couple[0])] = float(couple[1])
                else:
                    X[aux, int(couple[0])] = float(couple[1][:-2])



    elif name_real_dataset == 'ovarian'
        for line in g:
            line,data_line=line.split(",")[::-1],[]
            y.append(dict_type[str(line[0])])
            
            for aux in line[1:len(line)]:
                data_line.append(float(aux))
            X0=pd.concat([X0,pd.DataFrame(data_line).T])

#-------------UCI--------------------------
	
	elif name_real_dataset in ['dexter', 'gisette', 'madelon', 'arcene', 'farm-ads']:
	    X = np.zeros((N_real, P_real))
	    y = np.zeros(N_real)

	    aux = -1
	    for lines in train_data:
	        aux += 1
	        line = lines.split(" ")
	        
	        for j in range(len(line)):
	            if str(line[j])!='\n': #if line not ended
	                X[aux, j] = int(line[j])
	            else:
	                break
	                

	    for lines in valid_data:
	        aux += 1
	        line = lines.split(" ")

	        for j in range(len(line)):
	            if str(line[j])!='\n': #if line not ended
	                X[aux, j] = int(line[j])
	            else:
	                break

	    aux = -1
	    for lines in train_labels:
	        aux += 1
	        y[aux] = lines
	    for lines in valid_labels:
	        aux += 1
	        y[aux] = lines


	    if type_real_dataset > -1:             
	        X0.index = range(N)
	        X0.columns = range(P)
        X = X0.values



#----------------STANDARDIZE--------------------------
    
    N,P = X.shape
    print 'Shape: '+str((N, P))

    for i in range(P):
        X[:,i] -= np.mean(X[:,i])
        X[:,i] /= np.linalg.norm(X[:,i] + 1e-10)


    np.savetxt('../../../datasets/datasets_processed/'+name_real_dataset+'/X.txt', X)
    np.savetxt('../../../datasets/datasets_processed/'+name_real_dataset+'/y.txt', y)



