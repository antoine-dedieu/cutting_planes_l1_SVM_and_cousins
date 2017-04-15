import sys
from compare_methods_classification import *


NP_list = [(50,100),(50,200)]
k0 = 7
type_Sigma_list = [1, 2]

rho_tau  = [(0,1),   (0.2,1),   (0.5,1),   (0.8,1), 
			(0,0.7), (0.2,0.7), (0.5,0.7), (0.8,0.7), 
			(0,0.5), (0.2,0.5), (0.5,0.5), (0.8,0.5), 
			(0,0.3), (0.2,0.3), (0.5,0.3), (0.8,0.3)]

rho, tau = rho_tau[int(sys.argv[1])]


#CHECK TIME LIMIT AND K0max and NUMBER NS BEFORE RUNNING
#compare_methods_classification('squared_hinge', 50, 100, 5, float(rho), float(tau), 1)

compare_methods_classification('squared_hinge', 50, 100, 5, 0.8, 1, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0.8, 0.5, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0.8, 0.3, 1)

compare_methods_classification('squared_hinge', 50, 100, 5, 0.5, 1, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0.5, 0.3, 1)

compare_methods_classification('squared_hinge', 50, 100, 5, 0.2, 1, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0.2, 0.5, 1)

compare_methods_classification('squared_hinge', 50, 100, 5, 0, 1, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0, 0.5, 1)
compare_methods_classification('squared_hinge', 50, 100, 5, 0, 0.3, 1)


#for type_Sigma in type_Sigma_list:
    #for NP in NP_list:
        #N,P = NP
        #CHECK TIME LIMIT AND K0max and NUMBER NS BEFORE RUNNING
        #compare_heuristics(N,P,k0,float(rho),float(SNR),type_Sigma)