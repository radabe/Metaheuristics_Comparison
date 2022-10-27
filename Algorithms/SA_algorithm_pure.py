""" This script defines the Simmulated Annealing Algortihm. 
***
Its base functions and code were taken from: Mostapha Kalami Heris, Practical Genetic Algorithms in Python and MATLAB – Video Tutorial (URL: https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab), Yarpiz, 2020. 
and adapted for the Simmulated Annealing application and to the Engineering Change domain.
***
"""

# The SA Algorithm

def sa_run(problem, params):
    
    #Problem Information
    cf = problem['costfunc']
    st = problem['calc_stock']
    number_var = problem['nvar']
    vmin = problem['varmin']
    vmax = problem['varmax']+1
    no_parts = problem['parts']
    indep = problem['independency']
    best_init = problem['best_init']
    
    validity=0
    
    # Parameters
    maxit = params['maxit'] # number of iterations
    alpha = params['alpha'] # Evaporation rate [0.01, 0.2]
    temp_term = params['temp_term'] # Evaporation rate [0.01, 0.2]
    temp = params['temp'] #
    sigma = params['sigma']# distance searchspace
    
    # Empty Individual Template
    empty_individual = {}
    empty_individual = {'position':None,'cost':None, 'stock':None}
    empty_individual['position']=np.ones((1,number_var))
    
    # Initialize best solution
    bestsol = copy.deepcopy(empty_individual)
    bestsol['cost']= np.inf
    
    #generate initial random solution
    validity=0
    init_sol=copy.deepcopy(empty_individual)
    while validity < 1:
        validity=0
    init_sol['position'] = best_init
    init_sol['stock'] = st(init_sol['position'])
    init_sol['cost']= cf(init_sol['position'])

        for j in range (0, no_parts):
            if (init_sol['stock'][j]<0):
                validity -= 1
           else:
                validity += 1

        validity/= no_parts
    
    if init_sol['cost']<bestsol['cost']:
        bestsol=copy.deepcopy(init_sol)
        
    # Best Cost of Iterations
    bestcost = np.empty(maxit)
     
    #initialize last solution
    last_sol=init_sol
    
    #main loop
    for it in range(maxit):
        
        #initialize candidate
        candidate = copy.deepcopy(empty_individual)
        
        for n in range(temp_term):
            
            validity=0
            while validity < 1:
                #initialize candidate
                candidate['position']= last_sol['position']+ np.random.randint(-sigma, sigma,size=last_sol['position'].shape)
                candidate=apply_bound(candidate,vmin,vmax)
                
                #set all effectivity dates to same value
                if indep == 0:
                    for j in range(number_var):
                        if ((j%3==1)&(j>2)):
                            candidate['position'][j]=candidate['position'][j-int(number_var/no_parts)]

                #check if solution feasible
                validity=0
                candidate['stock']=st(candidate['position'])
                for m in range (0, no_parts):

                    if candidate['stock'][m]<0:
                        validity -= 1
                    else:     
                        validity += 1 
                validity/= no_parts
            
            #update costs
            candidate['cost']=cf(candidate['position'])
            
            #difference between current and last solution
            diff = candidate['cost']- last_sol['cost']
            
            #calculate metropolis acceptance criterion (random value whterh to accept new soluiton)
            metropolis = np.exp(-diff/temp)
            
            #update best solution
            if candidate['cost']<bestsol['cost']:
                bestsol=copy.deepcopy(candidate)
            
            #update current solution
            if candidate['cost']<last_sol['cost']:
                last_sol = copy.deepcopy(candidate)
            else:
                if np.random.rand() < metropolis:
                    last_sol = copy.deepcopy(candidate)
        
        #calculate temparture
        temp*=alpha
        
        # Store Best Cost
        bestcost[it]=bestsol['cost']

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}, parameter = {}, stock = {}".format(it, bestcost[it], bestsol['position'], bestsol['stock']))
            
    # Output
    out = {}
    out['bestsol'] = bestsol
    out['bestcost'] = bestcost
    
    return out

def apply_bound (x, vmin, vmax):
    x['position'] = np.maximum(x['position'], vmin) #replace invalid value with minimum possible value
    x['position'] = np.minimum(x['position'], vmax-1) #replace invalid value with maximum possibe value
    return x

def roulette_wheel_selection(n):
    c = np.cumsum(n)
    r = sum(n)*np.random.rand()
    ind = np.argwhere(r<=c)
    return ind[0][0]

