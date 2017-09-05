from compare_L1_SVM_both_CG_CP import *
from compare_L1_SVM_both_CG_CP_path import *

type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(1000,500),(1000, 1000), (1000,2000), (2000,2000), (1000,5000), (2000,5000), (1000,10000), (2000,10000)], 7, 0.2, 1
#type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(10000,10000), (10000,50000), (20000,50000)], 7, 0.1, 10

type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(10000,50000)], 7, 0.1, 10


compare_L1_SVM_both_CG_CP(type_Sigma, N_P_list, k0, rho, tau_SNR)