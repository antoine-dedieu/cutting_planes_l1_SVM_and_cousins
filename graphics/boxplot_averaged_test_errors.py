import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pylab as Py
from matplotlib.patches import Polygon



def boxplot_averaged_test_errors(fixed_random, metric_to_plot, accuracy_sparsity):

    fig = plt.figure(figsize=(10,5))
    ax  = fig.add_subplot(1,1,1)
    bp  = ax.boxplot(metric_to_plot, 0, '')

    colors = {0:'r', 1:'g', 2:'b', 3:'#FFA500', 4:'#FFA500', 5:'#FFA500', 6:'#FFA500'}


#---Boxplots
    for i in range(len(metric_to_plot)):
        box = bp['boxes'][i]

    #---Color inside of the box
        boxX = []
        boxY = []
        for j in range(5): #elements of the box
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))

        boxPolygon = Polygon(boxCoords, facecolor=colors[i])
        ax.add_patch(boxPolygon)

    #---Median lines
        med     = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax.plot(medianX, medianY, lw =3, c='k')


#---Labels
    dict_accuracy_sparsity_labels = {'accuracy' :('L1+L0', 'L2+L0', 'L0', 'Lasso', 'LP', 'E Net', 'Ridge'), 
                                     'sparsity':('L1+L0', 'L2+L0', 'L0', 'Lasso', 'LP', 'E Net')}
    ax.set_xticklabels(dict_accuracy_sparsity_labels[accuracy_sparsity])


    for ticks in [ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()]:
        for tick in ticks:
                tick.label.set_fontsize(16) 

    ax.set_ylim(bottom=0)   
    dict_accuracy_sparsity_title = {'accuracy': 'Prediction accuracy with '+str(fixed_random)+' design', 
                                    'sparsity': 'Sparsity with '+str(fixed_random)+' design'}
            
    ax.set_title(dict_accuracy_sparsity_title[accuracy_sparsity], fontsize=20,loc='center')


