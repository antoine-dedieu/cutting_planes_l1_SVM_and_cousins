import sys
import os
import datetime
import random
import numpy as np

sys.path.append('../graphics')
from boxplot_averaged_metrics import *



def aggregate():

	current_path = os.path.dirname(os.path.realpath(__file__))

#---SIMULATIONS
	metrics_to_average  = []

	n_average = 10
	for i in range(n_average):
		metrics  = np.load(current_path+'/l2_estimation/'+str(i)+'/misclassification/metrics.npy')
		metrics_to_average.append(metrics)


	for aux_metric in range(4):
		name_metric     = ['l2_estimation', 'misclassification', 'sparsity', 'true_positive'][aux_metric]
		metric_averaged = np.array([ metrics_to_average[i][aux_metric] for i in range(n_average) ]) 

		boxplot_averaged_metrics(metric_averaged,  name_metric)
		plt.savefig(current_path+'/simulations/'+name_metric+'.pdf')


aggregate()




