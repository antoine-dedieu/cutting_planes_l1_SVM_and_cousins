import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pylab as Py
Py.style.use('ggplot')


import numpy as np
from scipy import interpolate
import time


def AUC(beta, X, y):
    idx_plus  = np.where(y == 1)[0]
    idx_minus = np.where(y == -1)[0]

    AUC_score = 0

    for i in idx_plus:
        for j in idx_minus:
            AUC_score += int(np.dot(beta, X[i, :] -X[j, :]) >0)

    AUC_score /= float(len(idx_plus)*len(idx_minus))
    return AUC_score



def AUC(beta, X, y):
    idx_plus  = np.where(y == 1)[0]
    idx_minus = np.where(y == -1)[0]

    AUC_score = 0

    for i in idx_plus:
        for j in idx_minus:
            AUC_score += int(np.dot(beta, X[i, :] -X[j, :]) >0)

    AUC_score /= float(len(idx_plus)*len(idx_minus))
    return AUC_score


def statistical_graphics_real_datasets(type_loss, X_train, X_test, y_train, y_test, betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1, K0_list, N,P):

    #K0_SUBLIST : VALUES OF K0 WE STUDY
    #BETAS_BENCHMARK : BETAS FOR ALGORITHM 1 WITH ONE ITERATIONS

    #NO PLOT
    plt.ioff()

    N_train,P = X_train.shape
    N_test,P = X_test.shape


    #Initialize K0_sublist with size of real support
    K0_sublist = []
    

    train_errors = [  [[1e10] for K0 in K0_list] for i in range(4) ]
    test_errors = [  [[1e10] for K0 in K0_list] for i in range(4) ]
    AUC_errors = [  [[0] for K0 in K0_list] for i in range(4) ]
    

    
#---Figure
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    color = {0:'r',1:'g',2:'b',3:'c',4:'m'}
    color_bis = {0:'r--',1:'g--',2:'b--',3:'c--',4:'m--'}
    


#---MAIN LOOP
    var_max, var_min = 0, 1000
    legend = ['H_L2','H_L1', 'RFE_L2', 'RFE_L1']
    
    K0_min = K0_list[0]
    aux = -1

    for betas in [betas_Heuristic_L2, betas_Heuristic_L1, betas_RFE_L2, betas_RFE_L1]:
        aux += 1
        betas_non_null = [[np.zeros(P)] for K0 in K0_list]

        for K0 in K0_list:

            for loop in range(len(betas[K0])):
                #print betas[K0]
                beta, beta0 = betas[K0][loop]

            #---Compute errors
                if len(np.where(beta!=0)[0]) == K0:
                    dot_product_y_train = y_train*(np.dot(X_train,beta)+ beta0*np.ones(N_train))
                    dot_product_y_test = y_test*(np.dot(X_test,beta)+ beta0*np.ones(N_test))

                    dict_loss_train = {'hinge': np.sum([max(0, 1-dot_product_y_train[i])  for i in range(N_train)]),
                                       'squared_hinge': np.sum([max(0, 1-dot_product_y_train[i])**2  for i in range(N_train)]) }

                    dict_loss_test = {'hinge': np.sum([max(0, 1-dot_product_y_test[i])  for i in range(N_test)]),
                                      'squared_hinge': np.sum([max(0, 1-dot_product_y_test[i])**2  for i in range(N_test)]) }

                #---Train and test
                    train_errors[aux][K0-K0_min].append(dict_loss_train[type_loss])
                    test_errors[aux][K0-K0_min].append(dict_loss_test[type_loss])
                    betas_non_null[K0-K0_min].append(beta)


                #---AUC
                    AUC_score = AUC(beta, X_test, y_test)
                    AUC_errors[aux][K0-K0_min].append(AUC_score)

                    
                    
        print '\n NEW MODEL'
        print [round(np.min(test_errors[aux][k]),2) for k in range(len(K0_list))]
        #print [np.argmin(test_errors[aux][k]) for k in range(len(K0_list))]
        
        tab = [np.argmin(test_errors[aux][k]) for k in range(len(K0_list))]
        
        for k in range(len(K0_list)):
            print round(train_errors[aux][k][ tab[k] ], 3), np.where(betas_non_null[k][ tab[k] ]!=0)[0]
        
        K0_sublist.append(np.argmin([np.min(test_errors[aux][k]) for k in range(len(K0_list))]))
        
        
        
#---If K0_sublist has less than 4 elements we add some k0 by using the mean and std of the one we have
    old_K0_sublist = np.copy(K0_sublist)
    
    K0_sublist =list(set(K0_sublist))
    mean, std = np.mean(K0_sublist), np.std(K0_sublist)
    
    K0_sublist.append(int(mean))
    K0_sublist.append(int(mean-std))
    K0_sublist.append(int(mean+std))

    K0_sublist = list(set(K0_sublist))
    K0_sublist = sorted(K0_sublist[:4])
    
    
#---PLOT RESULTS
    for aux in range(4):
        
        
        if aux<2:
            color_idx=-1
            
            for K0 in K0_sublist:
                color_idx += 1
                
                dict_color = {0: color, 1: color_bis}
                train_error, test_error = train_errors[aux][K0-K0_min][1:], test_errors[aux][K0-K0_min][1:]
                AUC_error = AUC_errors[aux][K0-K0_min][1:]
                
                if len(train_error)>1:
                    f = interpolate.interp1d(train_error, test_error)
                    ax1.plot(train_error, f(train_error), dict_color[aux][color_idx%5],label= str(legend[aux])+' K0='+str(K0),lw=1)
                    
                    f = interpolate.interp1d(train_error, AUC_error)
                    ax2.plot(train_error, f(train_error), dict_color[aux][color_idx%5], label= str(legend[aux])+' K0='+str(K0),lw=1)
                    

             
                         
        else:
            dict_markers = {2: 'v', 3: 'o'}
            idx_list = np.array([np.argmin(test_errors[aux][K0-K0_min]) for K0 in K0_sublist])
            
            train_error, test_error = [], []
            AUC_error, variable_selection = [], []
            
            for i in range(len(K0_sublist)):
                if idx_list[i]>0:
                    
                    train_error.append(train_errors[aux][ K0_sublist[i]-K0_min ][ idx_list[i] ]) 
                    test_error.append( test_errors[aux][ K0_sublist[i]-K0_min ][ idx_list[i] ])
                    AUC_error.append(  AUC_errors[aux][ K0_sublist[i]-K0_min ][ idx_list[i] ])
                    
            
            ax1.scatter(train_error[0], test_error[0], s=40, c='k', marker=dict_markers[aux] , label= str(legend[aux])+' for K0 with best lambda')
            ax2.scatter(train_error[0], AUC_error[0], s=40, c='k', marker=dict_markers[aux] , label= str(legend[aux])+' for K0 with best lambda')
            
            
            for i in range(len(train_error)):
                ax1.scatter(train_error[i], test_error[i], s=40, c='k', marker=dict_markers[aux])
                ax2.scatter(train_error[i], AUC_error[i], s=40, c='k', marker=dict_markers[aux])
                
                ax1.annotate(str(K0_sublist[i]), (train_error[i], test_error[i]) ,size=15)
                ax2.annotate(str(K0_sublist[i]), (train_error[i], AUC_error[i]) ,size=15)


#---PARAMETERS


#---AX1
    ax1.set_ylabel('Test error', fontsize=16)
    
    test_min = np.min([np.min([np.min(test_errors[aux][k]) for k in range(len(K0_list))]) for aux in range(4)])
    ax1.set_ylim(0, 10*test_min)
    
    ax1.set_title('Test error with Train error for '+str(type_loss)+ ' loss'+' \nN='+str(N)+'; P='+str(P), 
             fontsize=18,loc='center')
    
#---AX2
    ax2.set_ylabel('Area under the curve', fontsize=16)
    
    ax2.set_title('AUC with Train error for '+str(type_loss)+ ' loss'+' \nN='+str(N)+'; P='+str(P), 
             fontsize=18,loc='center')


    
#---GENERAL

    for legend in [ax1.legend(loc=7), ax2.legend(loc=7)]:
        frame = legend.get_frame()
        frame.set_facecolor('0.7')
    
    mpl.rcParams['figure.facecolor'] = '0.75'
    mpl.rcParams['grid.color'] = 'black'


    

    
    