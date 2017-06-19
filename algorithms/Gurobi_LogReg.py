import numpy as np
from gurobipy import *
import datetime


def Gurobi_LogReg_model(type_penalization, X, y, K, alpha, betas_linearize, range_x, log_loss, grad_loss, constraint_in_model):

#idx_CG         = index of the features used to generate 
#BETA_LINEARIZE = required -> linearization around 

#range_x, log_loss, grad_loss : parameters for constraints
    
    beta_linearize, beta_0_linearize = betas_linearize[0], betas_linearize[1]
    norm_beta_linearize = np.linalg.norm(beta_linearize)
    
#---MODEL and VARIABLES
    Gurobi_LogReg=Model("Gurobi_LogReg")
    Gurobi_LogReg.setParam('OutputFlag', True )
    Gurobi_LogReg.setParam('TimeLimit', 300)

    
    #Linearized log loss
    xi = np.array([Gurobi_LogReg.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

    betas_plus  = np.array([Gurobi_LogReg.addVar(lb=0, name="beta_+_"+str(j)) for j in range(P)])
    betas_minus = np.array([Gurobi_LogReg.addVar(lb=0, name="beta_-_"+str(j)) for j in range(P)])
    b0 = Gurobi_LogReg.addVar(lb=-GRB.INFINITY, name="b0")


    #0-1 variables
    u = np.array([Gurobi_LogReg.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="u_"+str(j)) for j in range(P)])

    Gurobi_LogReg.update()


#---OBJECTIVE VALUE 
    dict_penalization = {'l1': quicksum(  betas_plus[j]+betas_minus[j]                               for j in range(P)),
                         'l2': quicksum( (betas_plus[j]+betas_minus[j])*(betas_plus[j]+betas_minus[j]) for j in range(P))}

    Gurobi_LogReg.setObjective(quicksum(xi) + alpha*dict_penalization[type_penalization], GRB.MINIMIZE)


#---HYPERPLAN CONSTRAINTS
    how_far = 1./norm_beta_linearize*(y*np.dot(X, beta_linearize))

    for i in range(N):
        if len(constraint_in_model[i]) == 0:
            Gurobi_LogReg.addConstr(xi[i]>= 0, name="slack_"+str(i))
        else:
            for x in constraint_in_model[i]:
                Gurobi_LogReg.addConstr(xi[i]- log_loss[x] + grad_loss[x]*( quicksum( y[i]*X[i][j]*(betas_plus[j] - betas_minus[j]) for j in range(P) )  
                                                                            + y[i]*b0 - range_x[x] )>= 0, 
                                        name="slack_"+str(i)+"axis_"+str(x))


#---SPARSITY CONSTRAINTS
    M= 2*np.max(np.abs(beta_linearize))

    for j in range(P):
        Gurobi_LogReg.addConstr(betas_plus[j] + betas_minus[j] <= M*u[j], name="sparsity_"+str(j))

    Gurobi_LogReg.addConstr(quicksum(u) <= K, name="degree_sparsity")
    


#---WARM START
    for j in range(P):
        betas_plus[j].start  = max( beta_linearize[j], 0)
        betas_minus[j].start = max(-beta_linearize[j], 0)
    b0.start = beta_0_linearize

    return Gurobi_LogReg





def constraint_generation_logreg(type_penalization, X, y, K, alpha, betas_linearize):

    epsilon_RC = 1e-3

#---Constraints parameters
    max_linearize = 0.1 #range to linearize
    N_constraints = 100
    
    beta_linearize, beta_0_linearize = betas_linearize[0], betas_linearize[1]
    norm_beta_linearize = np.linalg.norm(beta_linearize)

    range_x   = np.linspace(-norm_beta_linearize*max_linearize, norm_beta_linearize*max_linearize, N_constraints)
    log_loss  = np.array([math.log(1 + math.exp(-x)) for x in range_x])
    grad_loss = np.array([1./(1 + math.exp(x))       for x in range_x])


#---Constraints index
    constraint_index     = np.round(np.linspace(0, N_constraints-1, N_constraints/10)).astype(int).tolist() 
    constraint_not_index = list(set(range(N_constraints))-set(constraint_index))
    
    constraint_in_model = []
    constraint_to_check = []
    how_far = 1./norm_beta_linearize*(y*np.dot(X, beta_linearize))
    
    for i in range(N):
        if how_far[i] > max_linearize:
            print i
            constraint_in_model.append([])
            constraint_to_check.append(range(N_constraints))
        else:
            constraint_in_model.append(constraint_index)
            constraint_to_check.append(constraint_not_index)

    Gurobi_LogReg = Gurobi_LogReg_model(type_penalization, X, y, K, alpha, betas_linearize, range_x, log_loss, grad_loss, constraint_in_model)

    
#---Infinite loop
    continue_loop = True
    n_round       = 0
    
    while continue_loop:
        n_round += 1
        continue_loop = False
        write_and_print('Round '+str(n_round), f)

    #---Optimize
        Gurobi_LogReg.update()
        Gurobi_LogReg.optimize()


    #---Parameters
        xi          = [Gurobi_LogReg.getVarByName('loss_'+  str(i)) for i in range(N)]

        betas_plus  = [Gurobi_LogReg.getVarByName('beta_+_'+str(j)) for j in range(P)]
        betas_minus = [Gurobi_LogReg.getVarByName('beta_-_'+str(j)) for j in range(P)]
        beta        = np.array([beta_plus.X  for beta_plus  in betas_plus]) - np.array([beta_minus.X for beta_minus in betas_minus])

        b0          = Gurobi_LogReg.getVarByName('b0')
        beta_0      = b0.X 


    #---Constraint Generation
        RC_aux = y_train*(np.dot(X_train, beta) + beta_0)

        for i in range(N):
        #---Check all reduced costs 
            RC_array   = log_loss[constraint_to_check[i]] - grad_loss[constraint_to_check[i]]*(RC_aux[i] - range_x[constraint_to_check[i]])
            RC_argmax  = np.argmax(RC_array)
            RC_max     = RC_array[RC_argmax]
            idx_to_add = constraint_to_check[i][RC_argmax]

        #---Add most violated constraint
            if RC_max - xi[i].X > epsilon_RC:
                continue_loop = True
                print 'Constraint: '+str(i)+' index: '+str(idx_to_add)
                
                Gurobi_LogReg.addConstr(xi[i]- log_loss[idx_to_add] + grad_loss[idx_to_add]*( quicksum( y[i]*X[i][j]*(betas_plus[j] - betas_minus[j]) for j in range(P) )  
                                                                                                      + y[i]*b0 - range_x[idx_to_add] ) >= 0, 
                                            name="slack_"+str(i)+"axis_"+str(idx_to_add))
                constraint_in_model[i].append(idx_to_add)
                constraint_to_check[i].remove(idx_to_add)
                
                
    #---Upper bound
        print '##### Upper bound: '+str(np.sum([math.log(1 + math.exp(-RC_aux[i])) for i in range(N)]) + alpha*np.sum(np.abs(beta)))
        print '##### Lower bound: '+str(Gurobi_LogReg.ObjVal)+'\n\n'
        print beta
    
    return xi, beta, beta_0, Gurobi_LogReg.ObjVal





