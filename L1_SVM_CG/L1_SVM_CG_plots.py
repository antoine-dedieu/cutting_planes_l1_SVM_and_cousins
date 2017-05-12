import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def L1_SVM_CG_plots(type_Sigma, N, P_list, k0, rho, tau_SNR, times_L1_SVM, times_penalizedSVM_R, times_SAM_R_SVM, times_SVM_CG):
   

    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(1,1,1)
    
    #P_max = P_list[::-1][0]
    #bp1 = ax1.boxplot(times_L1_SVM, 0, '', positions = range(len(P_list)), widths = P_max/20)
    #bp2 = ax1.boxplot(times_SVM_CG, 0, '', positions = range(len(P_list)), widths = P_max/20) 

    positions = np.arange(0.5, 0.5+len(P_list), 1)

    bp1 = ax1.boxplot(times_L1_SVM,         0, '', positions = positions, widths = len(P_list)/50.)
    bp2 = ax1.boxplot(times_penalizedSVM_R, 0, '', positions = positions, widths = len(P_list)/50.)
    bp3 = ax1.boxplot(times_SAM_R_SVM,      0, '', positions = positions, widths = len(P_list)/50.)
    bp4 = ax1.boxplot(times_SVM_CG,         0, '', positions = positions, widths = len(P_list)/50.)
    bps = [bp1, bp2, bp3, bp4]
    
    

 #--COLORS IN THE BOXES
    colors = {0:'y', 1:'b', 2:'r', 3:'g'}
    legend_plot = {0:'Gurobi', 1:'R penalized SVM', 2:'R SAM', 3:'Gurobi CG'}
    
    for i in range(4):
        bp = bps[i]
        
        for P in range(len(P_list)):
            box = bp['boxes'][P]

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
            med = bp['medians'][P]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                print medianX, medianY
                ax1.plot(medianX, medianY, lw =3, c='k')

        #---Legend
            if P == 0:
                box.set_label(legend_plot[i])
                box.set_color(colors[i])


#---LABELS
    ax1.set_xticks(positions)
    #ax1.set_xticklabels(('L1', 'L1', 'L2', 'L2', 'L2XB', 'L2XB', 'L0', 'Lasso', 'LP', 'Ridge'))
    ax1.set_xticklabels((str(P) for P in P_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; N='+str(N)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of features', fontsize=18)
    ax1.set_ylabel('Time', fontsize=18)

    ax1.set_xlim(left=0)
    
    legend = ax1.legend()
    for label in legend.get_texts():
        label.set_fontsize('x-large')
    #ax1.set_xlim(right=1.1*P_max)






def L1_SVM_CG_plots_path(type_Sigma, N, P_list, k0, rho, tau_SNR, times_L1_SVM_averaged, times_SVM_CG_averaged_delete, times_SVM_CG_averaged_no_delete):


    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(1,1,1)

    positions = np.arange(0.5, 0.5+len(P_list), 1)

    bp1 = ax1.boxplot(times_L1_SVM_averaged,                0, '', positions = positions, widths = len(P_list)/50.)
    bp2 = ax1.boxplot(times_SVM_CG_averaged_delete,         0, '', positions = positions, widths = len(P_list)/50.)
    bp3 = ax1.boxplot(times_SVM_CG_averaged_no_delete,      0, '', positions = positions, widths = len(P_list)/50.)
    bps = [bp1, bp2, bp3]
    
    

 #--COLORS IN THE BOXES
    colors = {0:'b', 1:'r', 2:'g'}
    #legend_plot = {0:'No CG', 1:'CG with correlation, Eps=1e-2', 2:'Liblinear eps = 1e-2'}
    legend_plot = {0:'No CG', 1:'CG with correlation, Eps=1e-1', 2:'CG with correlation, Eps=1e-2'}
    
    for i in range(3):
        bp = bps[i]
        
        for P in range(len(P_list)):
            box = bp['boxes'][P]

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
            med = bp['medians'][P]
            medianX = []
            medianY = []
            for j in range(2):
                medianX.append(med.get_xdata()[j])
                medianY.append(med.get_ydata()[j])
                print medianX, medianY
                ax1.plot(medianX, medianY, lw =3, c='k')

        #---Legend
            if P == 0:
                box.set_label(legend_plot[i])
                box.set_color(colors[i])

    

#---LABELS
    ax1.set_xticks(positions)
    ax1.set_xticklabels((str(P) for P in P_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; N='+str(N)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of features', fontsize=18)
    ax1.set_ylabel('Time', fontsize=18)

    ax1.set_xlim(left=0)
    
    legend = ax1.legend()
    for label in legend.get_texts():
        label.set_fontsize('x-large')


def compare_ratios(type_Sigma, N, P_list, k0, rho, tau_SNR, compare_ratio, alpha_list):
   

    fig = plt.figure(figsize=(30,20))
    ax1 = fig.add_subplot(1,1,1)
    
    
    for P in range(len(P_list)):
        plt.plot(np.arange(len(alpha_list)), compare_ratio[P,:], lw=2, label=[P_list[P]])


    
#---LABELS
    ax1.set_xticks(range(len(alpha_list)))
    ax1.set_xticklabels((str(alpha) for alpha in alpha_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 
            
    ax1.set_title('Sigma='+str(type_Sigma)+'; N='+str(N)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Alpha', fontsize=18)
    ax1.set_ylabel('Average ratio of time improvements', fontsize=18)


    legend = ax1.legend(loc=7)
    frame = legend.get_frame()


    ax1.set_xlim(left=0)





def L1_SVM_plots_errorbar(type_Sigma, arg_list, k0, rho, tau_SNR, times_list, legend_plot, time_or_objval):


    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,1,1)

    positions = np.arange(0.5, 0.5+len(arg_list), 1)
    

    n_to_plot   = len(times_list)
    loop_repeat = len(times_list[0])



#--arguments
    colors      = {0:'r', 1:'g', 2:'b', 3:'#FFA500', 4:'m'}
    markers     = {0:'.', 1:'+', 2:'*', 3:'D', 4:'.'}
    linestyles  = {0:':', 1:'-', 2:'-.', 3:'--', 4:'-.'}
    markersizes = {0:'15', 1:'15', 2:'10', 3:'8', 4:'15'}

    for i in range(n_to_plot):
        mean_method = np.mean(times_list[i], axis=1)
        std_method  = np.std(times_list[i], axis=1)

        ax1.errorbar(positions, mean_method, yerr=std_method, fmt='-o', color=colors[i], label=legend_plot[i], lw=3, marker=markers[i], linestyle=linestyles[i], markersize=markersizes[i])
        #ax1.errorbar(positions, mean_method, yerr=std_method, fmt='-o', color=colors[i], label=legend_plot[i])

    

#---LABELS
    ax1.set_xticks(positions)
    ax1.set_xticklabels((str(arg) for arg in arg_list))

    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(14) 
            
    #ax1.set_title('Sigma='+str(type_Sigma)+'; N='+str(N)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('Number of features', fontsize=18)

    if time_or_objval == 'time':
        ax1.set_ylabel('Time (s)', fontsize=18)
    elif time_or_objval == 'objval':
        ax1.set_ylabel('Objective values ratio ', fontsize=18)

    ax1.set_xlim(left=0)
    

    legend = ax1.legend(loc=2)
    for label in legend.get_texts():
        label.set_fontsize('xx-large')























