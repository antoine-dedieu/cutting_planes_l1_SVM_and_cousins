import sys
from compare_methods_classification import *


type_Sigma = 4

rho_dmu_Sigma2  = [(.2,.5), (.2,1), (.2,3), (.2,5), 
				   (.5,.5), (.5,1), (.5,3), (.5,5)]
rho_dmu_Sigma4  = [(.2,1), (.2,2), (.2,5), (.2,10), 
				   (.5,1), (.5,2), (.5,5), (.5,10)]
dict_rho_dmu = {2: rho_dmu_Sigma2, 4: rho_dmu_Sigma4}

rho, d_mu = dict_rho_dmu[type_Sigma][int(sys.argv[1])]


n_average = 1
average_simulations_compare_methods_classification('hinge', 50, 100, 7, rho, d_mu, type_Sigma, n_average)
