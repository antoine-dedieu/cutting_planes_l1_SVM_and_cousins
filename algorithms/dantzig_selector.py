def dantzig_selector(X, y, k, start_beta, time_limit):

#INPUT
#X, y       : inputs
#k          : sparisty constraint
#beta_start : possible warm-start -> if no warm_start then use []
#time_limit : for Gurobi
    
#OUTPUT
#beta_dantzig
#objective_value
    
    
#---MODEL
    N,P = X.shape
    Dantzig = Model("MIO")
    Dantzig.setParam('OutputFlag', False )
    Dantzig.setParam('TimeLimit', time_limit)
    
    
    
#---VARIABLES and CONSTRAINTS
    beta     = np.array([Dantzig.addVar(lb=-GRB.INFINITY, name="beta_"+str(i)) for i in range(P)])
    z        = [Dantzig.addVar(0.0, 1.0, 1.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)] 
    grad_inf = Dantzig.addVar(name="grad_inf")

    Dantzig.update()
        

#---SPARISTY CONSTRAINT
    M=20
    for i in range(P):
        Dantzig.addConstr( beta[i] <= M*z[i], "max_beta_"+str(i))
        Dantzig.addConstr(-beta[i] <= M*z[i], "min_beta_"+str(i))
    
    Dantzig.addConstr(quicksum(z) <= k, "sparsity")
    
    
#---INFINITE NORM CONSTRAINTS
    for i in range(P):
        Dantzig.addConstr( quicksum(X[k,i]*y[k] for k in range(N)) 
                          -quicksum(  quicksum(X[l,i]*X[l,k] for l in range(N)) * beta[k] for k in range(P)) 
                          <= grad_inf, name='Dantzig_sup_'+str(k) )
        
        Dantzig.addConstr(- quicksum(X[k,i]*y[k] for k in range(N)) 
                          + quicksum(  quicksum(X[l,i]*X[l,k] for l in range(N)) * beta[k] for k in range(P)) 
                          <= grad_inf, name='Dantzig_inf_'+str(k) )
    

#---OBJECTIVE VALUE    
    obj = grad_inf

    
#---WARM START
    if len(start_beta)>0:
        for i in range(P):
            beta[i].start = start_beta[i]
            z[i].start = int(start_beta[i]!=0)
            
    
#---SOLVE
    Dantzig.setObjective(obj, GRB.MINIMIZE)
    Dantzig.optimize()
    
    beta_Dantzig = [beta[i].x for i in range(P)]
    obj_value    = Dantzig.ObjVal
    
    return beta_Dantzig, obj_value