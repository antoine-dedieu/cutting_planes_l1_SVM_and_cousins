import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pylab as Py
from matplotlib.patches import Polygon




def bars_test_errors(fixed_random, metric_to_plot, accuracy_sparsity):
    
#metric_to_plot: of size 

    plt.ioff() #no plot
    fig = plt.figure(figsize=(10,5))
    ax  = fig.add_subplot(1,1,1)
    
    colors ={0:'r', 1:'g', 2:'b', 3:'#FFA500', 4:'#FFA500', 5:'#FFA500', 6:'#FFA500'}
    
#---Plots
    width = 0.5
    dict_accuracy_sparsity_idx    = {'accuracy':0, 'sparsity':1}
    dict_accuracy_sparsity_length = {'accuracy':7, 'sparsity':6} #no ridge for sparsity

    for i in range(dict_accuracy_sparsity_length[accuracy_sparsity]):
        rects = ax.bar(i-width/4, metric_to_plot[i][dict_accuracy_sparsity_idx[accuracy_sparsity]], width/2, color=colors[i])        


#---Labels
    ax.set_xticks(range(dict_accuracy_sparsity_length[accuracy_sparsity]))

    dict_accuracy_sparsity_labels = {'accuracy':('L1+L0', 'L2+L0', 'L0', 'Lasso', 'LP', 'E Net', 'Ridge'), 
                                     'sparsity':('L1+L0', 'L2+L0', 'L0', 'Lasso', 'LP', 'E Net')}
    ax.set_xticklabels(dict_accuracy_sparsity_labels[accuracy_sparsity])


    for ticks in [ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(16) 

    ax.set_ylim(bottom=0)

    dict_accuracy_sparsity_title = {'accuracy': 'Prediction accuracy for '+str(fixed_random)+' design', 
                                    'sparsity': 'Sparsity for '+str(fixed_random)+' design'}
            
    ax.set_title(dict_accuracy_sparsity_title[accuracy_sparsity], fontsize=20,loc='center')












