from compare_L1_SVM_CP      import *
from compare_L1_SVM_CP_path import *

type_Sigma, N_list, P, k0, rho, tau_SNR = 1, [10000, 50000, 100000], 100, 7, 0.2, 1

compare_L1_SVM_CP(type_Sigma, N_list, P, k0, rho, tau_SNR)