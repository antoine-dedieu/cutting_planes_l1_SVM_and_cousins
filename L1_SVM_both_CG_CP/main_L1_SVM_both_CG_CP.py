from compare_L1_SVM_both_CG_CP import *
from compare_L1_SVM_both_CG_CP_path import *

type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(1000,500),(1000, 1000), (1000,2000), (2000,2000), (1000,5000), (2000,5000), (1000,10000), (2000,10000)], 7, 0.2, 1
type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(1000,2000), (2000,2000), (2000,5000), (2000,10000), (2000,15000)], 7, 0.2, 1
#type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(500,1000), (500,2000), (500,5000), (500,10000), (500,15000)], 10, 0.1, 1
type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(1000,1000), (1000,2000), (1000,5000), (1000,10000), (1000,20000), (1000,50000), (1000,100000)], 10, 0.1, 1
type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(500,1000), (500,2000), (500,5000), (500,10000), (500,20000), (500,50000), (500,100000)], 10, 0.1, 1

type_Sigma, N_P_list, k0, rho, tau_SNR = 2, [(500,20000), (500,50000), (500,100000)], 10, 0.1, 1

#type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(1000,200), (2000,200), (2000,500)], 7, 0.2, 1


#type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(200,200), (500,500)], 7, 0.2, 1
#type_Sigma, N_P_list, k0, rho, tau_SNR = 1, [(200,200), (500,500)], 7, 0.5, 1

compare_L1_SVM_both_CG_CP(type_Sigma, N_P_list, k0, rho, tau_SNR)