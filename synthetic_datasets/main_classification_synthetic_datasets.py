import sys
from compare_methods_classification import *


type_Sigma = 4

rho_dmu_Sigma2  = [(.2,.5), (.2,.8), (.2,1), (.2, 1.5),
				   (.5,.8), (.5,1), (.5,1.5), (.5,2)]
rho_dmu_Sigma4  = [(.2,1), (.2,2), (.2,5), (.2,10), 
				   (.5,1), (.5,2), (.5,5), (.5,10)]
dict_rho_dmu = {2: rho_dmu_Sigma2, 4: rho_dmu_Sigma4}

rho, d_mu = dict_rho_dmu[type_Sigma][int(sys.argv[1])]


n_average = 10
average_simulations_compare_methods_classification('logreg', 50, 100, 7, rho, d_mu, type_Sigma, n_average)
#average_simulations_compare_methods_classification('logreg', 100, 10000, 7, 0.2, 10, 4, n_average)
