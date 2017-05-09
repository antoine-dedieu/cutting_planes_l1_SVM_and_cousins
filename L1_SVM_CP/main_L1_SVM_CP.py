from compare_L1_SVM_CP      import *
from compare_L1_SVM_CP_path import *

type_Sigma, N_list, P, k0, rho, tau_SNR = 1, [200, 500, 1000, 2000, 5000], 200, 7, 0.2, 1
#type_Sigma, N_list, P, k0, rho, tau_SNR = 1, [100, 200, 500, 1000], 100, 7, 0.5, 1

compare_L1_SVM_CP(type_Sigma, N_list, P, k0, rho, tau_SNR)