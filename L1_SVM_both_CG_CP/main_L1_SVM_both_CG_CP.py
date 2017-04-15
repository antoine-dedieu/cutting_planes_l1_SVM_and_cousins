from compare_L1_SVM_both_CG_CP import *
from compare_L1_SVM_both_CG_CP_path import *

type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(200,200), (500,500), (1000, 1000), (2000, 2000)], 7, 0.2, 1
#type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(200,200), (500,500)], 7, 0.2, 1
#type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(200,200), (500,500)], 7, 0.5, 1

compare_L1_SVM_both_CG_CP_path(type_Sigma, N_P_list, k0, rho, tau_SNR)