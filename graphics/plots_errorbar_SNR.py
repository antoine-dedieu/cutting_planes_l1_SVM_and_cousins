import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

#mpl.rcParams['text.usetex'] = True




def plots_errorbar_SNR(SNR_list, metric_averaged, legends, name_metric):

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,1,1)

#--arguments
    colors      = {0:'g', 1:'b', 2:'#FFA500', 3:'#FFA500'}
    markers     = {0:'.', 1:'+', 2:'*', 3:'D', 4:'.'}
    linestyles  = {0:':', 1:'-', 2:'-.', 3:'--', 4:'-.'}
    markersizes = {0:'15', 1:'15', 2:'10', 3:'8', 4:'15'}

    positions = np.arange(0.5, 0.5+len(SNR_list), 1)

    for i in range(len(metric_averaged)):
        mean_method = np.mean(metric_averaged[i], axis=1)
        std_method  = np.std(metric_averaged[i],  axis=1)
        print mean_method, std_method

        ax1.errorbar(positions, mean_method, yerr=std_method, fmt='-o', color=colors[i], label=legends[i], lw=3, marker=markers[i], linestyle=linestyles[i], markersize=markersizes[i])
        #max_y_list.append(int(100*np.max(mean_method)) )
        #ax1.errorbar(positions, mean_method, yerr=std_method, fmt='-o', color=colors[i], label=legend_plot[i])


#---LABELS
    ax1.set_xticks(positions)
    ax1.set_xticklabels((str(arg) for arg in SNR_list))
    
    for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(14) 
            
    #ax1.set_title('Sigma='+str(type_Sigma)+'; N='+str(N)+'; SNR='+str(tau_SNR)+'; Rho='+str(rho)+'\n', fontsize=20,loc='center')
    ax1.set_xlabel('SNR', fontsize=18)
    ax1.set_ylabel(name_metric, fontsize=18)
    ax1.set_xlim(left=0)
    

    legend = ax1.legend(loc=2)
    for label in legend.get_texts():
        label.set_fontsize('xx-large')























