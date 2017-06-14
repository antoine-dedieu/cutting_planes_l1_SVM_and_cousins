import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


def compare_all_k(all_norm_list, K0_list, name_metric):

	fig    = plt.figure(figsize=(10,5))
	ax1    = fig.add_subplot(1,1,1)
	color  = {0:'g', 1:'b', 2:'#FFA500', 3:'#FFA500'}


#---Plot
	labels = {0:'L0+L1', 1:'L0+L2', 2:'L1 SVM', 3:'L2 SVM'}

	for i in range(2):
		dict_delta_x = {0:-0.1, 1:0.1}
		for K0 in K0_list[4:]:
			for metric in all_norm_list[i][K0]:
				ax1.scatter(K0 + dict_delta_x[i], metric,  10, color[i], marker='o')
			ax1.scatter(K0 + dict_delta_x[i], np.min(all_norm_list[i][K0]), 30, color[i], marker='s')
		ax1.scatter(K0 + dict_delta_x[i], np.min(all_norm_list[i][K0]), 30, color[i], marker='s', label=labels[i])

		
	K0_max = K0_list[::-1][0]

	for i in [2,3]:
		for metric in all_norm_list[i]:
			ax1.scatter(K0_max+i, metric, 10, color[i], marker='o')
		ax1.scatter(K0_max+i, np.min(all_norm_list[i]), 30, color[i], marker='s', label=labels[i])


#---Labels
	ax1.set_xticks(range(2, K0_max +5))
	ax1.set_xticklabels(range(2, K0_max+1)+['','L1','L2',''])
		
	for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
		for tick in ticks:
			tick.label.set_fontsize(14) 

	ax1.set_xlim(left=2)

	dict_name_metric = {'l2_estimation': 'L2 estimation', 'misclassification': 'Misclassification'}
	ax1.set_ylabel(dict_name_metric[name_metric], fontsize=16)
	#ax1.grid()

	legend = ax1.legend(loc=2, fontsize=12, framealpha=1)
	frame = legend.get_frame()
	frame.set_facecolor('1')


	ax1.set_title(dict_name_metric[name_metric]+' with train error', fontsize=18,loc='center')






def compare_best_k(list_betas_K0, name_metric):


	fig    = plt.figure(figsize=(10,5))
	ax1    = fig.add_subplot(1,1,1)
	color  = {0:'g', 1:'b', 2:'k--', 3:'k'}
	labels = {0:'L1, K0='+str(list_betas_K0[0][2]), 
			  1:'L2, K0='+str(list_betas_K0[1][2]), 
			  2:'L1 SVM', 
			  3:'L2 SVM'}

	for i in range(4):
		ax1.plot(list_betas_K0[i][0], list_betas_K0[i][1], color[i], lw=3, label=labels[i])
		

	

	legend = ax1.legend(loc=2, fontsize=14, framealpha=1)
	frame = legend.get_frame()
	frame.set_facecolor('1')


	for ticks in [ax1.xaxis.get_major_ticks(), ax1.yaxis.get_major_ticks()]:
		for tick in ticks:
			tick.label.set_fontsize(14) 



	dict_name_metric = {'l2_estimation': 'L2 estimation', 'misclassification': 'Misclassification'}
	ax1.set_ylabel(dict_name_metric[name_metric], fontsize=16)
	ax1.set_xlabel('Train error', fontsize=16 )

	ax1.set_title(dict_name_metric[name_metric]+' with train error', fontsize=18,loc='center')



