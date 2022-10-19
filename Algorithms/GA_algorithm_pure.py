#!/usr/bin/env python
# coding: utf-8

# # The Genetic Algorithm

# In[1]:


def ga_run (problem, params):
    
    #Problem Information (from input file)
    cf = problem['costfunc']
    st = problem['calc_stock']
    prob_var = problem['nvar']
    vmin = problem['varmin']
    
    #add 1 to ensure full solution space
    vmax = problem['varmax']+1
    no_parts = problem['parts']
    no_var= int(prob_var/no_parts)
    indep = problem['independency']
    init_pop = problem['init_pop']
    validity=0
    
    # Parameters
    maxit = params['maxit'] # number of iterations
    npop = params['npop'] #number of population
    pc = params['pc'] # ratio children / parents
    nc = int(np.round(pc*npop/2)*2) # children
    gamma = params['gamma']
    mu= params['mu']
    sigma = params ['sigma']
    beta = params['beta']
    
    # Empty Individual Template
    empty_individual = {}
    empty_individual = {'position':None,'cost':None, 'stock':None}
    empty_individual['position']=np.zeros((1,prob_var), dtype=int)
    
    # BestSolution
    bestsol = copy.deepcopy(empty_individual)
    bestsol['cost']= np.inf
    
    # Initialize Population
    pop = [None]*npop
    for n in range (0, npop):
        pop[n] = empty_individual
        pop[n]['position']=init_pop[n]
       
    for i in range (0, npop):
    #    validity=0
    #    while validity < 1:
    #        for j in range (prob_var):
    #            #check if all on one day or on multiple days
    #            if indep == 0:
    #                if ((j%3==1) & (j>2)):
    #                    pop[i]['position'][0][j]=pop[i]['position'][0][j-no_var]
    #                else:
    #                    pop[i]['position'][0][j]= np.random.randint(vmin,vmax)
    #            else:
    #                pop[i]['position'][0][j]= np.random.randint(vmin,vmax)
    #            
        pop[i]['stock'] = st(pop[i]['position'])
             
    #        for j in range (0, no_parts):
    #            if (pop[i]['stock'][j]<0):
    #                validity -= 1
    #            else:
    #                validity += 1
    #        
    #        validity/= no_parts
    #    if validity==1:
        pop[i]['cost']= cf(pop[i]['position'])
        if pop[i]['cost']<bestsol['cost']:
            bestsol=copy.deepcopy(pop[i])
        
    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    # Main Loop
    for it in range(maxit):
        
        costs = np.array([x['cost'] for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)
        
        popc = []
        for k in range(nc//2):
            # Select Parents
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]'
            
            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            
            # Perform Crossover
            c1,c2 = crossover (p1, p2, gamma)
            
            #Round Values to integer
            c1 = rounding(c1)
            c2 = rounding(c2)
            
            # Perform Mutation
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)
            
            # Apply Bound
            apply_bound(c1, vmin, vmax)
            apply_bound(c2, vmin, vmax)
            
            #setting every effectivity date the same
            if indep == 0:
                for j in range(0,prob_var):
                    if ((j%3==1) & (j>2)):
                        c1['position'][j]=c1['position'][j-no_var]
                        c2['position'][j]=c2['position'][j-no_var]
            
            # Evaluate First Offspring and append to popc if valid
            c1['cost']=cf(c1['position'])
            c1['stock']=st(c1['position'])
            for j in range (0, no_parts):
                if c1['stock'][j]<0:
                    validity= validity-1
            if validity >= 0:
                if c1['cost'] < bestsol['cost']:
                    bestsol = copy.deepcopy(c1)
                    popc.append(c1)
            validity = 0
                
            # Evaluate Second Offspring
            c2['cost']=cf(c2['position'])
            c2['stock']=st(c2['position'])
            for j in range (0, no_parts):
                if c2['stock'][j]<0:
                    validity= validity-1
            if validity >= 0:
                if c2['cost'] < bestsol['cost']:
                    bestsol = copy.deepcopy(c2)
                    popc.append(c2)
            validity = 0
                  
            # Add Offspring to popc
            
            
        # Merge, Sort and Select new pop
        pop = pop + popc
        pop = sorted(pop, key=lambda x: x['cost'])
        pop = pop[0:npop]

        # Store Best Cost
        bestcost[it]=bestsol['cost']

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}, parameter = {}, stock = {}".format(it, bestcost[it], bestsol['position'], bestsol['stock']))
            
    # Output
    out = {}
    out['pop'] = pop
    out['bestsol'] = bestsol
    out['bestcost'] = bestcost
    return out


# In[2]:


def crossover(p1, p2, gamma):
    c1= copy.deepcopy(p1)
    c2= copy.deepcopy(p2)
    alpha = np.random.uniform(-gamma,1+gamma,*c1['position'].shape)
    c1['position']=alpha*p1['position']+(1-alpha)*p2['position']
    c2['position']=alpha*p2['position']+(1-alpha)*p1['position']
    return c1, c2


# In[3]:


def mutate(x, mu, sigma):
    y = copy.deepcopy(x)
    ind = np.argwhere(np.random.rand(*x['position'].shape) <= mu) #find index where x smaller then the mutation factor
    y['position'][ind] = y['position'][ind] + np.random.randint(-sigma,sigma, size=ind.shape)
    return y


# In[4]:


def apply_bound (x, vmin, vmax):
    x['position'] = np.maximum(x['position'], vmin) #replace invalid value with minimum possible value
    x['position'] = np.minimum(x['position'], vmax-1) #replace invalid value with maximum possibe value
    return x


# In[5]:


def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r<=c)
    return ind[0][0]


# In[6]:


def rounding(x):
    x['position'] = np.around(x['position'])
    return x


# In[7]:


def check_buildability():
    return

