ddef L1_SVM_CG_model_AND_relax(l0_OR_not, X, idx_CG, y, K, alpha, time_limit, model):

    
#l0_OR_not = L0 or NOT L0 -> if yes we add constraints
#idx_CG    = index of the features used to generate 
#MODEL     = previous model to speed up
    

#---DEFINE A NEW MODEL IF NO PREVIOUS ONE
    N,P  = X.shape
    P_CG = len(idx_CG)

    
    if(model == 0):
    
    #---VARIABLES
        L1_SVM_CG=Model("L1_SVM_CG")
        L1_SVM_CG.setParam( 'OutputFlag', False )
        L1_SVM_CG.setParam('TimeLimit', time_limit)
        
        
        #Hinge loss
        xi = np.array([L1_SVM_CG.addVar(lb=0, name="loss_"+str(i)) for i in range(N)])

        #Beta -> name correspond to real index
        beta_plus  = np.array([L1_SVM_CG.addVar(lb=0, name="beta_+_"+str(idx_CG[i])) for i in range(P_CG)])
        beta_minus = np.array([L1_SVM_CG.addVar(lb=0, name="beta_-_"+str(idx_CG[i])) for i in range(P_CG)])
        b0 = L1_SVM_CG.addVar(lb=-GRB.INFINITY, name="b0")
        
        L1_SVM_CG.update()


    #---OBJECTIVE VALUE 
        L1_SVM_CG.setObjective(quicksum(xi) + alpha*quicksum(beta_plus[i]+beta_minus[i] for i in range(P_CG)), GRB.MINIMIZE)


    #---HIGE CONSTRAINTS
        for i in range(N):
            L1_SVM_CG.addConstr(xi[i] + y[i]*(b0 + quicksum([ X[i][idx_CG[k]]*(beta_plus[k] - beta_minus[k]) for k in range(P_CG)]))>= 1, 
                                 name="slack_"+str(i))
      
    #---RELAX
        L1_SVM_CG.update()
        L1_SVM_CG_relax = L1_SVM_CG.copy()
        
    
    
    
#------L0 CONSTRAINT OR NOT
        if l0_OR_not=='l0':
            
            M=5*K
            
        #---MODEL
        
            #Define z for all P
            z = [L1_SVM_CG.addVar(0.0, 1.0, 0.0, GRB.BINARY, name="z_"+str(i)) for i in range(P)]
            L1_SVM_CG.update()
            
            for i in range(P_CG):
                L1_SVM_CG.addConstr(M*z[idx_CG[i]] - beta_plus[i] - beta_minus[i] >= 0, "big_M_constraint_"+str(idx_CG[i]))
            for col in set(range(P))-set(idx_CG):
                L1_SVM_CG.addConstr(M*z[col] >= 0, "big_M_constraint_"+str(col))
                
            L1_SVM_CG.addConstr(quicksum(z) <= K, "sparsity") #sparsity
            
        
        #---RELAX
            
            beta_plus_relax  = [L1_SVM_CG_relax.getVarByName('beta_+_'+str(idx_CG[i])) for i in range(P_CG)]
            beta_minus_relax = [L1_SVM_CG_relax.getVarByName('beta_-_'+str(idx_CG[i])) for i in range(P_CG)]
            
            for i in range(P_CG):
                L1_SVM_CG_relax.addConstr(- beta_plus_relax[i] - beta_minus_relax[i] >= -M, "big_M_constraint_"+str(idx_CG[i]))  #L1
            
            L1_SVM_CG_relax.addConstr( -quicksum(beta_plus_relax[i]+beta_minus_relax[i] for i in range(P_CG)) >= -M*K, "L1") #L infinite
            
        
        
#---IF PREVIOUS MODEL JUST UPDATE THE PENALIZATION
    else:
        print 0        

    L1_SVM_CG.update()
    L1_SVM_CG_relax.update()
    
    return L1_SVM_CG, L1_SVM_CG_relax




def add_var(L1_SVM_CG, L1_SVM_CG_relax, most_violated_column, K):
# Add the most violated column in both model with respective constraints
    
    for grb in [L1_SVM_CG, L1_SVM_CG_relax]:
        
        col_plus, col_minus = Column(), Column()
        slack_constraints = [grb.getConstrByName('slack_'+str(i)) for i in range(N)]
        for i in range(N):
            col_plus.addTerms(  y_train[i]*X_train[i][most_violated_column], slack_constraints[i])
            col_minus.addTerms(-y_train[i]*X_train[i][most_violated_column], slack_constraints[i])

            
    #---MODEL
        if grb == L1_SVM_CG:
            big_M_constraint  = grb.getConstrByName('big_M_constraint_'+str(most_violated_column))
            col_plus.addTerms( -1, big_M_constraint)
            col_minus.addTerms(-1, big_M_constraint)
        
        
    #---RELAX
        else:
            L1 = grb.getConstrByName('L1')
            col_plus.addTerms( -1, L1)
            col_minus.addTerms(-1, L1)
        
        
    #---ADD IN BOTH CASES
        beta_plus  = grb.addVar(lb=0, obj = alpha, column=col_plus,  name="beta_+_"+str(most_violated_column) )
        beta_minus = grb.addVar(lb=0, obj = alpha, column=col_minus, name="beta_-_"+str(most_violated_column) )
        
        
    #---ADD THE MISSING CONSTRAINT FOR THE RELAX PROBLEM
        if grb == L1_SVM_CG_relax:
            grb.update()
            grb.addConstr(- beta_plus - beta_minus >= -5*K**2, name="big_M_constraint_"+str(most_violated_column))
            
        grb.update()
    










start = time.time()

K, alpha = 7, 0.5
time_limit = 60

model, _ = L1_SVM_CG_model_AND_relax('l0', X_train, range(P), y_train, K, alpha, time_limit, 0)

model.optimize()
beta_plus = [model.getVarByName('beta_+_'+str(i)).X for i in range(P)]
beta_minus = [model.getVarByName('beta_-_'+str(i)).X for i in range(P)]
beta = np.round(np.array(beta_plus) - np.array(beta_minus),6)
support = np.where(beta!=0)[0]


print 'Time: '+str(time.time()-start)

print '\nLen support = '+str(len(support))
print 'Support = '+str(support)
print 'Train error = '+str(model.ObjVal)











#---Initialize with RFE
estimator = svm.LinearSVC(penalty='l2', loss= 'squared_hinge', dual=True, C=1/float(2*alpha))
selector = RFE(estimator, n_features_to_select=15, step=1)
selector = selector.fit(X_train, y_train)
index_CG = np.where(selector.ranking_ < 2)[0].tolist()


#---Initialize the model
start = time.time()
model, model_relax = L1_SVM_CG_model_AND_relax('l0', X_train, index_CG, y_train, K, alpha, time_limit, 0)
aux = 0


#---Infinite loop until all the variables have non reduced cost

while True:  
    aux += 1
    
#---Solve the relax problem to get the dual solution
    model_relax.optimize()
    
    dual_slack = [model_relax.getConstrByName('slack_'+str(i)).Pi for i in range(N)]
    
    #print 'Slack: '+str(dual_slack)

    
    
#---Compute all reduce cost and look for most violated negative cost in the relax problem
    most_violated_cost   = 0
    most_violated_column = -1
    
    for column in set(range(P))-set(index_CG):
        reduced_cost = np.sum([y_train[i]*X_train[i,column]*dual_slack[i] for i in range(N)])
        reduced_cost = alpha  + min(reduced_cost, -reduced_cost)

        if reduced_cost <0:
            count +=1
            index.append(column)
        
        if reduced_cost < most_violated_cost:
            most_violated_cost   = reduced_cost
            most_violated_column = column
    
    
#---Add the column with the most most violated constraint to the original model (not the relax !!)
    if most_violated_column>=0:
        add_var(model, model_relax, most_violated_column, K) #add for both model
        index_CG = sorted(index_CG+[most_violated_column])

    else:
        break

print 'Time: '+str(time.time()-start)
print 'Number of rounds: '+str(aux)
        
        
#---Keep the K highest elements of the relax model
beta_plus_relax, beta_minus_relax = np.zeros(P), np.zeros(P)
beta_plus_relax[index_CG]   = [model_relax.getVarByName('beta_+_'+str(i)).X for i in index_CG]
beta_minus_relax[index_CG]  = [model_relax.getVarByName('beta_-_'+str(i)).X for i in index_CG]
beta0 = model_relax.getVarByName('b0')

beta = np.array(beta_plus_relax) - np.array(beta_minus_relax)
support = np.where(beta != 0)[0]

if len(support) > K:
    idx_start = sorted(np.abs(beta).argsort()[::-1][:K])
else:
    idx_start = support

beta_start = np.zeros(P)
beta_start[idx_start] = beta[idx_start]
print 'Start support = '+str(idx_start)
print 'Start train error = '+str(get_Objective('hinge', 'l1', X_train, beta_start, beta_0, y_train))
    


#---Warm-start the model
beta_plus_warm_start  = [model.getVarByName('beta_+_'+str(i)) for i in idx_start]
beta_minus_warm_start = [model.getVarByName('beta_-_'+str(i)) for i in idx_start]
z = [model.getVarByName('z_'+str(i)) for i in idx_start]


for i in range(len(idx_start)):
    beta_plus_warm_start[i].start  = beta_plus_relax[idx_start[i]]
    beta_minus_warm_start[i].start = beta_minus_relax[idx_start[i]]
    z[i].start = 1


model.update()

#---Finally solve the MIP
model.optimize()

beta_plus, beta_minus = np.zeros(P), np.zeros(P)
beta_plus[index_CG]  = [model.getVarByName('beta_+_'+str(i)).X for i in index_CG]
beta_minus[index_CG] = [model.getVarByName('beta_-_'+str(i)).X for i in index_CG]
beta = np.array(beta_plus) - np.array(beta_minus)
support = np.where(beta!=0)[0]


print '\nTime: '+str(time.time()-start)

print 'Len support = '+str(len(support))
print 'Support = '+str(support)
print 'Train error = '+str(model.ObjVal)





def get_Objective(type_loss, type_penalization, X, beta, beta_0, y):
    
    N = len(y)
    dot_product_y = y*(np.dot(X,beta)+ beta_0*np.ones(N))

    dict_loss = {'hinge': np.sum([max(0, 1-dot_product_y[i])  for i in range(N)]),
                 'squared_hinge': np.sum([max(0, 1-dot_product_y[i])**2  for i in range(N)]) }

    dict_penalization = {'l1': np.sum(np.abs(beta)),
                         'l2': np.linalg.norm(beta)**2}

    error = dict_loss[type_loss] + alpha*dict_penalization[type_penalization]
    return error








