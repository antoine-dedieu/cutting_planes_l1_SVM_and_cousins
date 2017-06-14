#import matplotlib.pyplot as plt
import numpy as np

from bars_test_errors import *


def plot_and_store(min_test_errors_l1_l2_l0, min_test_errors_Lasso_Ridge_Enet, pathname, fixed_random, title):

	min_test_errors_all_penalization  = min_test_errors_l1_l2_l0  + min_test_errors_Lasso_Ridge_Enet

	bars_test_errors(fixed_random, min_test_errors_all_penalization, 'accuracy')
	plt.savefig(pathname+'/accuracy_'+fixed_random+'_design_bars_after_'+title+'.pdf')
	plt.close()

	bars_test_errors(fixed_random, min_test_errors_all_penalization, 'sparsity')
	plt.savefig(pathname+'/sparsity_'+fixed_random+'_design_bars_after_'+title+'.pdf')
	plt.close()

	np.save(pathname+'/min_test_errors_'+fixed_random+'_all_penalization',  min_test_errors_all_penalization)