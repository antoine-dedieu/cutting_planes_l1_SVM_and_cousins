from compare_L1_SVM_CG      import *
from compare_L1_SVM_CG_path import *


#type_Sigma, N, P_list, k0, rho, tau_SNR  = 2, 100, [1000000], 10, 0.1, 1
#type_Sigma, N, P_list, k0, rho, tau_SNR = 2, 100, [1000, 2000, 5000, 10000, 20000, 50000, 100000], 10, 0.1, 1
#type_Sigma, N, P_list, k0, rho, tau_SNR = 2, 100, [1000, 2000, 5000, 10000, 20000, 50000, 100000], 10, 0.1, 1

type_Sigma, N, P_list, k0, rho, tau_SNR = 2, 100, [500, 1000], 10, 0.2, 1



compare_L1_SVM_CG(type_Sigma, N, P_list, k0, rho, tau_SNR)
#compare_L1_SVM_CG_path(type_Sigma, N, P_list, k0, rho, tau_SNR)