import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import pylab as Py
from matplotlib.patches import Polygon



def boxplot_averaged_metrics(metric_to_plot, name_metric):

    fig = plt.figure(figsize=(10,5))
    ax  = fig.add_subplot(1,1,1)

    colors ={0:'g', 1:'b', 2:'#FFA500', 3:'#FFA500'}

    dict_accuracy_sparsity_length = {'l2_estimation':4, 'misclassification':4, 'sparsity':3, 'true_positive':3}
    number_plots = dict_accuracy_sparsity_length[name_metric]

    print metric_to_plot

    #new_metric_to_plot = []
    #for j in range(len(metric_to_plot)):
    #    new_metric_to_plot.append( metric_to_plot[j][:number_plots].tolist() )

#---Boxplots
    medianprops = dict(linewidth=2.5, color='k')
    bp  = ax.boxplot(metric_to_plot, 0, '', medianprops=medianprops)

    for i in range(number_plots):
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
    ax.set_xticks(range(1, number_plots+1))

    dict_accuracy_sparsity_labels = {'l2_estimation'    :('L1+L0', 'L2+L0', 'L1', 'L2'), 
                                     'misclassification':('L1+L0', 'L2+L0', 'L1', 'L2'),
                                     'sparsity'         :('L1+L0', 'L2+L0', 'L1'),
                                     'true_positive'    :('L1+L0', 'L2+L0', 'L1')}
    ax.set_xticklabels(dict_accuracy_sparsity_labels[name_metric])
    ax.set_xlim(right = number_plots+0.5)


    for ticks in [ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()]:
        for tick in ticks:
            tick.label.set_fontsize(14) 

    ax.set_title(name_metric, fontsize=18,loc='center')



