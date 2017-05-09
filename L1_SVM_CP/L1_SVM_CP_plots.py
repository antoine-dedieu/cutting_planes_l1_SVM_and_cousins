import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np


def L1_SVM_CP_plots(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM, times_penalizedSVM_R, times_SAM_R_SVM, times_SVM_CP):
   

    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(1,1,1)
    
    #P_max = P_list[::-1][0]
    #bp1 = ax1.boxplot(times_L1_SVM, 0, '', positions = range(len(P_list)), widths = P_max/20)
    #bp2 = ax1.boxplot(times_SVM_CG, 0, '', positions = range(len(P_list)), widths = P_max/20) 

    positions = np.arange(0.5, 0.5+len(N_list), 1)

    bp1 = ax1.boxplot(times_L1_SVM,         0, '', positions = positions, widths = len(N_list)/50.)
    bp2 = ax1.boxplot(times_penalizedSVM_R, 0, '', positions = positions, widths = len(N_list)/50.)
    bp3 = ax1.boxplot(times_SAM_R_SVM,      0, '', positions = positions, widths = len(N_list)/50.)
    bp4 = ax1.boxplot(times_SVM_CP,         0, '', positions = positions, widths = len(N_list)/50.)
    bps = [bp1, bp2, bp3, bp4]
    
    

 #--COLORS IN THE BOXES
    colors = {0:'y', 1:'b', 2:'r', 3:'g'}
    legend_plot = {0:'Gurobi', 1:'R penalized SVM', 2:'R SAM', 3:'Gurobi CG'}
    
    for i in range(4):
        bp = bps[i]
        
        for N in range(len(N_list)):
            box = bp['boxes'][N]

        #---Color inside of the box
            boxX = []
            boxY = []
            for j in range(5): #elements of the box
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))

            boxPolygon = Polygon(boxCoords, facecolor=colors[i])
            ax1.add_patch(boxPolygon)

        #---Median lines
            med = bp['medians'][N]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                print medianX, medianY
                ax1.plot(medianX, medianY, lw =3, c='k')

        #---Legend
            if N == 0:
                box.set_label(legend_plot[i])
                box.set_color(colors[i])

    

#---LABELS
    ax1.set_xticks(positions)
    #ax1.set_xticklabels(('L1', 'L1', 'L2', 'L2', 'L2XB', 'L2XB', 'L0', 'Lasso', 'LP', 'Ridge'))
    ax1.set_xticklabels((str(N) for N in N_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; P='+str(P)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of samples', fontsize=18)
    ax1.set_ylabel('Time', fontsize=18)

    ax1.set_xlim(left=0)
    legend = ax1.legend()
    for label in legend.get_texts():
        label.set_fontsize('x-large')










def L1_SVM_CP_plots_path(type_Sigma, N_list, P, k0, rho, tau_SNR, times_L1_SVM_averaged, times_SVM_CP_averaged_delete, times_SVM_CP_averaged_no_delete):


    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(1,1,1)

    positions = np.arange(0.5, 0.5+len(N_list), 1)

    bp1 = ax1.boxplot(times_L1_SVM_averaged,                0, '', positions = positions, widths = len(N_list)/50.)
    bp2 = ax1.boxplot(times_SVM_CP_averaged_delete,         0, '', positions = positions, widths = len(N_list)/50.)
    bp3 = ax1.boxplot(times_SVM_CP_averaged_no_delete,      0, '', positions = positions, widths = len(N_list)/50.)
    bps = [bp1, bp2, bp3]
    
    

 #--COLORS IN THE BOXES
    colors = {0:'b', 1:'r', 2:'g'}
    legend_plot = {0:'Gurobi', 1:'L1 norm', 2:'Liblinear SH L1'}
    
    for i in range(3):
        bp = bps[i]
        
        for N in range(len(N_list)):
            box = bp['boxes'][N]

        #---Color inside of the box
            boxX = []
            boxY = []
            for j in range(5): #elements of the box
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))

            boxPolygon = Polygon(boxCoords, facecolor=colors[i])
            ax1.add_patch(boxPolygon)

        #---Median lines
            med = bp['medians'][N]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                print medianX, medianY
                ax1.plot(medianX, medianY, lw =3, c='k')

        #---Legend
            if N == 0:
                box.set_label(legend_plot[i])
                box.set_color(colors[i])

    


#---LABELS
    ax1.set_xticks(positions)
    ax1.set_xticklabels((str(N) for N in N_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; P='+str(P)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of features', fontsize=18)
    ax1.set_ylabel('Time', fontsize=18)

    ax1.set_xlim(left=0)
    
    legend = ax1.legend()
    for label in legend.get_texts():
        label.set_fontsize('x-large')







def L1_SVM_CP_plots_errorbar_path(type_Sigma, N_list, P, k0, rho, tau_SNR, times_method_0, times_method_1, times_method_2, times_method_3, time_or_objval):


    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,1,1)

    positions = np.arange(0.5, 0.5+len(N_list), 1)
    

 #--COLORS IN THE BOXES
    colors = {0:'b', 1:'b--', 2:'g', 3:'r'}
    #legend_plot = {0:'No CG', 1:'CG with correlation, Eps=1e-2', 2:'Liblinear eps = 1e-2'}
    #legend_plot = {0:'No CG, no Warm start', 1:'No CG, Warm start', 2:'CG with correlation, eps=5e-1', 3:'CG with correlation, eps=1e-2'}`

    legend_plot = {0:'No CP, no Warm start', 1:'No CP, Warm start', 2:'CP with L1 norm, eps=1e-2', 3:'CP with liblinear, eps=1e-2'}

    loop_repeat = len(times_method_0)


    mean_method_0, std_method_0 = np.mean(times_method_0, axis=1), np.std(times_method_0, axis=1)
    mean_method_1, std_method_1 = np.mean(times_method_1, axis=1), np.std(times_method_1, axis=1)
    mean_method_2, std_method_2 = np.mean(times_method_2, axis=1), np.std(times_method_2, axis=1)
    mean_method_3, std_method_3 = np.mean(times_method_3, axis=1), np.std(times_method_3, axis=1)


    ax1.errorbar(positions, mean_method_0, yerr=std_method_0, fmt='-o', color='b', label=legend_plot[0])
    ax1.errorbar(positions, mean_method_1, yerr=std_method_1, fmt='-o', color='c', label=legend_plot[1])
    ax1.errorbar(positions, mean_method_2, yerr=std_method_2, fmt='-o', color='g', label=legend_plot[2])
    ax1.errorbar(positions, mean_method_3, yerr=std_method_3, fmt='-o', color='r', label=legend_plot[3 ])

    

#---LABELS
    ax1.set_xticks(positions)
    ax1.set_xticklabels((str(N) for N in N_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; P='+str(P)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of samples', fontsize=18)

    if time_or_objval == 'time':
        ax1.set_ylabel('Time (s)', fontsize=18)
    elif time_or_objval == 'objval':
        ax1.set_ylabel('Objective values ratio ', fontsize=18)

    ax1.set_xlim(left=0)
    

    legend = ax1.legend(loc=2)
    for label in legend.get_texts():
        label.set_fontsize('x-large')

    #legend = ax1.legend()
    #for label in legend.get_texts():
    #    label.set_fontsize('x-large')










