import numpy as np

def write_and_print(text,f):
    print text
    f.write('\n'+text)


def aux_results_K0(AUC_scores_size_K0, betas_support_size_K0, real_support, f):

# Return test results for fixed or random design
# fixed_random: string input

    argmax_AUC_error = np.argmax(AUC_scores_size_K0)
    max_AUC_error = AUC_scores_size_K0[argmax_AUC_error]

    beta_max = betas_support_size_K0[argmax_AUC_error]
    beta_max_support = set(np.where(beta_max!=0)[0])

    #Print
    write_and_print('Maximal AUC error of '+str(np.round(max_AUC_error,4)) ,f)
    write_and_print('Good variables : '+str(len( set(real_support) & set(beta_max_support))) ,f)  
    if len(beta_max_support)>0:
        write_and_print('Variable selection error : '+str(  len(beta_max_support-real_support) + len(real_support-beta_max_support)  ) ,f)

    write_and_print('\nBest support : '+ str(list(sorted(beta_max_support))) ,f)
    write_and_print('Real support : '+ str(list(real_support)) ,f)
    
    return max_AUC_error, beta_max





def aux_results_penalization(max_AUC_errors, max_betas, real_support, f):
    
# Give the two best validation K0 for every penalization 
# Return the two K0 and two corresponding test errors

    argmax_AUC_errors = np.argmax(max_AUC_errors)
    best_K0 = argmax_AUC_errors
    max_AUC_error = max_AUC_errors[argmax_AUC_errors]

    beta_max = max_betas[argmax_AUC_errors]
    beta_max_support = set(np.where(beta_max!=0)[0])


    if len(beta_max_support)>0:
        VS_error = len(beta_max_support-real_support) + len(real_support-beta_max_support)
    else:
        VS_error = len(real_support)

    write_and_print('\n\nBest K0 : '+str(best_K0)+'   Maximal AUC error of '+str(np.round(max_AUC_error,4)) ,f)
    write_and_print('Good variables : '+str(len( set(real_support) & set(beta_max_support))) ,f)  
    if len(beta_max_support)>0:
        write_and_print('Variable selection error : '+str(  len(beta_max_support-real_support) + len(real_support-beta_max_support)  ) ,f)

    write_and_print('\nBest support : '+ str(list(sorted(beta_max_support))) ,f)
    write_and_print('Real support : '+ str(list(real_support)) ,f)
    
    
    sorted_list = np.argsort(max_AUC_errors)[::-1][:2]
    top2_max_AUC_errors = [max_AUC_error, max_AUC_errors[sorted_list[1]]]

    return sorted_list.tolist(), top2_max_AUC_errors















