""" This script defines the Tabu Search Algortihm. 
***
Its base functions and code were taken from: Mostapha Kalami Heris, Practical Genetic Algorithms in Python and MATLAB – Video Tutorial (URL: https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab), Yarpiz, 2020. 
and adapted for the Tabu Search application and to the Engineering Change domain.
***
"""

# The TS Algorithm

def ts_run(problem, params):
    
    #Problem Information
    cf = problem['costfunc']
    st = problem['calc_stock']
    number_var = problem['nvar']
    vmin = problem['varmin']
    vmax = problem['varmax']+1
    no_parts = problem['parts']
    indep=problem['independency']
    best_init = problem['best_init']
    validity=0
        
    # Parameters
    maxit = params['maxit'] # number of iterations
    neighbors = params['npop'] #size of neigborhood [10,50]
    sigma = params['sigma']# distance searchspace
    
    # Empty Individual Template
    empty_individual = {}
    empty_individual = {'position':None,'cost':None, 'stock':None}
    empty_individual['position']=np.ones((1,number_var))
    empty_individual['cost']= np.inf
    
    # Initialize best solution
    bestsol = copy.deepcopy(empty_individual)
    bestsol['cost']= np.inf
    
    # Initialize initial solution
    init_sol=copy.deepcopy(empty_individual)
    
    #generate initial random solution
    validity=0
    
    while validity < 1:
       validity=0
       init_sol['position'] = np.random.randint(vmin,vmax, number_var)
       #set all effectivity dates to same value
       if indep == 0:
           for j in range(number_var):
               if ((j%3==1)&(j>2)):
                   init_sol['position'][j]=init_sol['position'][j-int(number_var/no_parts)]
       
       init_sol['stock'] = st(init_sol['position'])
       init_sol['cost']= cf(init_sol['position'])
       
       for j in range (0, no_parts):
           if (init_sol['stock'][j]<0):
               validity -= 1
           else:
               validity += 1
           
       validity/= no_parts
    
    init_sol['position'] = best_init
    init_sol['stock'] = st(init_sol['position'])
    init_sol['cost']= cf(init_sol['position'])
    
    if init_sol['cost']<bestsol['cost']:
        bestsol=copy.deepcopy(init_sol)
        
    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    #initialize tabu list
    tabu_list=[]
    tabu_list.append(init_sol['position'])
    
    #initialize last solution
    last_sol=init_sol
    
        
    #main loop
    for it in range(maxit):
    
        #initialize neighborhood
        neighborhood=[None]*neighbors
        best_neighbor=copy.deepcopy(empty_individual)
        best_neighbor['cost']= np.inf
        
        #create neighborhood of potential solutions
        for n in range(neighbors):
            neighborhood[n]=copy.deepcopy(empty_individual)
            neighborhood[n]['position']= last_sol['position']+ np.random.randint(-sigma, sigma,size=neighborhood[n]['position'].shape)
            neighborhood[n]=apply_bound(neighborhood[n],vmin,vmax)
            #toggle on off for effectivity date on single day
            for j in range(number_var):
                if indep == 0:
                    if ((j%3==1)&(j>2)):
                        neighborhood[n]['position'][0][j]=neighborhood[n]['position'][0][j-int(number_var/no_parts)]
            
            #check if solution feasible
            validity=0
            #print(neighborhood[n])
            neighborhood[n]['stock']=st(neighborhood[n]['position'][0])
            neighborhood[n]['cost']=cf(neighborhood[n]['position'][0])
            for m in range (0, no_parts):
                               
                if neighborhood[n]['stock'][m]<0:
                    validity -= 1
                else:     
                    validity += 1 
            validity/= no_parts
            
            if validity!=1:
                
                neighborhood[n]=copy.deepcopy(empty_individual)
        
        #filter out tabu list
        for t in range(len(tabu_list)):
            neighborhood=list(filter(lambda nb: type(nb) != int, neighborhood))
            for n in range(len(neighborhood)): 
                if (neighborhood[n]['position']==tabu_list[t]).all():
                    neighborhood[n]=copy.deepcopy(empty_individual)
        
        #print(neighborhood)
        #update costs
        for n in range(len(neighborhood)):
            if neighborhood[n]['cost']<bestsol['cost']:
                bestsol=copy.deepcopy(neighborhood[n])
            if neighborhood[n]['cost']<best_neighbor['cost']:
                best_neighbor=copy.deepcopy(neighborhood[n])
                
        #select next solution to explore (best solution of current neighborhood)
        last_sol = copy.deepcopy(best_neighbor)
        
        #update tabu_list
        if len(tabu_list)<(number_var*no_parts):
            tabu_list.append(last_sol['position'][0])
        else:
            del tabu_list[0]
            tabu_list.append(last_sol['position'][0])
        
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

