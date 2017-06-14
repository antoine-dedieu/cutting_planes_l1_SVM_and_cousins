import numpy as np

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt



def compare_all_k_real_datasets(min_test_errors_l1_l2_l0_all_K0, min_test_errors_Lasso_Ridge_Enet):
    
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,1,1)

    colors = {0:'r', 1:'g', 2:'b', 3:'#FFA500', 4:'#FFA500', 5:'#FFA500', 6:'#FFA500'}
    labels = {0:'L1', 1:'L2', 2:'L0'}
    
    range_k = range(2,20)
    
    L1_accuracy   = min_test_errors_l1_l2_l0_all_K0[0][2:20]
    L2_accuracy   = min_test_errors_l1_l2_l0_all_K0[1][2:20]
    L0_accuracy   = min_test_errors_l1_l2_l0_all_K0[2][2:20]

    
#---PLOTS
    min_test = np.min([np.min(L1_accuracy), np.min(L2_accuracy), np.min(L0_accuracy)])
    max_test = np.max([np.max(L1_accuracy), np.max(L2_accuracy), np.max(L0_accuracy)])

    width = 0.34
    size1=50
    size2=25
    
    [add1, add2, add3] = is_there_collision(L1_accuracy[0], L2_accuracy[0], L0_accuracy[0], width, min_test, max_test)

    rects1 = ax1.scatter(add1, L1_accuracy[0]  , s=size1, c=colors[0], marker='+', label='L1')
    rects1 = ax1.scatter(add2, L2_accuracy[0]  , s=size2, c=colors[1], label='L2')
    rects1 = ax1.scatter(add3, L0_accuracy[0]  , s=size1, c=colors[2], marker='*', label='L0')
    

    for i in range(1, len(range_k)):
        [add1, add2, add3] = is_there_collision(L1_accuracy[i], L2_accuracy[i], L0_accuracy[i], width, min_test, max_test)

        rects1 = ax1.scatter(i+add1, L1_accuracy[i]  , s=size1, c=colors[0], marker='+')
        rects1 = ax1.scatter(i+add2, L2_accuracy[i]  , s=size2, c=colors[1])
        rects1 = ax1.scatter(i+add3, L0_accuracy[i]  , s=size1, c=colors[2], marker='*')



    rects1 = ax1.scatter(i+2, min_test_errors_Lasso_Ridge_Enet[0][0], s=size2, color='k', label ='Lasso')
    rects1 = ax1.scatter(i+4, min_test_errors_Lasso_Ridge_Enet[1][0], s=size2, color='k', label ='E Net')
    rects1 = ax1.scatter(i+6, min_test_errors_Lasso_Ridge_Enet[2][0], s=size2, color='k', label ='Ridge')
    rects1 = ax1.scatter(i+8, min_test, s=1)
        
    
#---PARAMETERS
    ax1.set_xticks(range(len(range_k))+ [len(range_k)+1, len(range_k)+3, len(range_k)+5])
    ax1.set_xticklabels([range_k[i] for i in range(len(range_k))]+['Lasso','E Net', 'Ridge'])
    
    
#---Cut x and y
    ax1.set_ylim(bottom=0.5*min_test)
    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Prediction accuracy with k', fontsize=20,loc='center')

    ax1.set_xlim(left=0)

    legend = ax1.legend(loc=7, fontsize=14, framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('1')





def is_there_collision(L1_accuracy, L2_accuracy, L0_accuracy, width, min_y, max_y):
    threshold = 0.1 #approved

    arr = [L1_accuracy, L2_accuracy, L0_accuracy]

    [l1, l2, l3] = sorted(arr)
    argsort      = np.argsort(arr)

    a = abs(l1-l2) < threshold*(max_y - min_y)
    b = abs(l2-l3) < threshold*(max_y - min_y)

    if not a:
        if not b:
            add = [0,0,0]

        else:
            add = [0, -width/4, width/4] #add on y axis
            a,b = argsort[1], argsort[2] #changes in argsort to keep the order
            argsort[1] = min(a,b)
            argsort[2] = max(a,b)

    else:
        if not b:
            add = [-width/4, width/4, 0]
            a,b = argsort[0], argsort[1]
            argsort[0] = min(a,b)
            argsort[1] = max(a,b)

        else:
            argsort = [0,1,2]
            add     = [-width/2, 0, width/2]

    idx_position = np.argsort(argsort)
    result       = [add[idx_position[i]] for i in range(3)] 
    return result







