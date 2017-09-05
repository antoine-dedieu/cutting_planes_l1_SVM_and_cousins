import sys
from compare_methods_classification import *


rho_dmu_Sigma2  = [(.2,.5), (.2,.8), (.2,1), (.2, 1.5),
				   (.5,.8), (.5,1), (.5,1.5), (.5,2)]
rho_dmu_Sigma4  = [(.2,1), (.2,2), (.2,5), (.2,10), 
				   (.5,1), (.5,2), (.5,5), (.5,10)]
dict_rho_dmu = {2: rho_dmu_Sigma2, 4: rho_dmu_Sigma4}


dict_rho_NP = [(0,50, 100),   (.2,50, 100),   (.5,50, 100), 
			   (0,100, 200),  (.2,100, 200),  (.5,100, 200),
			   (0,100, 1000), (.2,100, 1000), (.5,100, 1000)]
				   

rho, N, P = dict_rho_NP[int(sys.argv[1])]


n_average = 3
#average_simulations_compare_methods_classification_with_SNR('logreg', N, P, 7, rho, 4, n_average)
average_simulations_compare_methods_classification('hinge', 10, 20, 7, 0.2, 10, 4, n_average)
