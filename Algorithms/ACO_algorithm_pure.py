""" This script defines the Ant Colony Optimization Algortihm. 

***
Its base functions and code were taken from: Mostapha Kalami Heris, Practical Genetic Algorithms in Python and MATLAB – Video Tutorial (URL: https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab), Yarpiz, 2020. 
and adapted for the Ant Colony Optimization application and to the Engineering Change domain.
***

"""

# The Ant Colony Optimisation Algorithm

def aco_run(problem, params):
    
    #Problem Information
    cf = problem['costfunc']
    st = problem['calc_stock']
    number_var = problem['nvar']
    vmin = problem['varmin']
    vmax = problem['varmax']+1
    no_parts = problem['parts']
    indep = problem['independency']
    init_pop = problem['init_pop']
    best_init = problem['best_init']
    validity=0
      
    # Parameters
    maxit = params['maxit'] # number of iterations
    alpha = params['alpha'] #Pheromone influence
    beta = params['beta'] #Heuristic influence
    rho = params['rho'] # Evaporation rate [0.01, 0.2]
    kappa = params['npop'] #Number of ants [10,50]
    
    # Empty Individual Template
    empty_individual = {}
    empty_individual = {'position':None,'cost':None, 'stock':None}
    
    # Initialise Pheromone Trails
    phero_matrix = np.zeros((number_var*vmax,number_var*vmax))
    diag=np.zeros((vmax,vmax))
    for k in range(1,vmax):
        diag+=np.eye(vmax,k=k)
    diag*=1/2
    
    for i in range(1, number_var):
        phero_matrix[vmax*(i-1):(vmax*i),(vmax)+vmax*(i-1):(vmax)+vmax*i]=1/2
        if (i % 3 == 2):
            phero_matrix[vmax*(i-1):(vmax*i),(vmax)+vmax*(i-1):(vmax)+vmax*i]-=diag
    
    #Initialise ants and update initial pheromone matrix
    ants = [None]*kappa
    for n in range (0, kappa):
        ants[n]= copy.deepcopy(empty_individual)
        ants[n]['position']=init_pop[n]
        ants[n]['cost']=cf(ants[n]['position'])
        for j in range(0,number_var-1):
            if ants[n]['cost']>0:
                phero_matrix[(j*vmax)+(int(ants[n]['position'][j]))][(j+1)*vmax+(int(ants[n]['position'][j+1]))]+=(1/ants[n]['cost'])
            else:
                phero_matrix[(j*vmax)+(int(ants[n]['position'][j]))][(j+1)*vmax+(int(ants[n]['position'][j+1]))]+=(1/0.1)

    # Define initial solution 
    init_sol = copy.deepcopy(empty_individual)
    init_sol['position']=best_init #insert random numbers
    init_sol['stock']=st(init_sol['position'])
    init_sol['cost']=cf(init_sol['position'])
    # Evaluate Initial Solution
    bestsol={}
    bestsol=copy.deepcopy(init_sol)                
    
    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    #Main Loop

    for it in range(maxit):
        
        for n in range(kappa):
            #select random starting position for each ant
            ants[n]['position']=np.zeros((1,number_var))
            ants[n]['position'][0][0]=np.random.randint(0,vmax)
             
            #select next value
            for j in range(1,number_var):
                #check if all on one day or on multiple days
                if indep == 0:
                    if ((j%3==1) & (j>2)):
                        ants[n]['position'][0][j]=ants[n]['position'][0][j-int((number_var/no_parts))]
                    else:
                        ants[n]['position'][0][j]=next_value(j-1,ants[n]['position'][0][j-1],phero_matrix,alpha,beta,n)
                else:
                    ants[n]['position'][0][j]=next_value(j-1,ants[n]['position'][0][j-1],phero_matrix,alpha,beta,n)
                
            #check if valid result
            ants[n]['stock']=st(ants[n]['position'][0])
            for m in range (0, no_parts):
                if ants[n]['stock'][m]<0:
                    validity = validity-1
            #if valid, calculate cost
            if validity >= 0:
                ants[n]['cost']=cf(ants[n]['position'][0])
                for j in range(0,number_var-1):              
                    #update pheromatrix online
                    #reinforcment phase
                    
                    if ants[n]['cost']>0:
                        phero_matrix[(j*vmax)+(int(ants[n]['position'][0][j]))][(j+1)*vmax+(int(ants[n]['position'][0][j+1]))]+=(1/ants[n]['cost'])
                    else:
                        phero_matrix[(j*vmax)+(int(ants[n]['position'][0][j]))][(j+1)*vmax+(int(ants[n]['position'][0][j+1]))]+=(1/0.1)
                if ants[n]['cost']<bestsol['cost']:
                    bestsol=copy.deepcopy(ants[n])
            validity = 0
                

                #add ants to list
            
        #pheromone update
        #evaporation
        phero_matrix *= (1-rho)
               
        # Store Best Cost
        bestcost[it]=bestsol['cost']
        
        # Show Iteration Information
        print("Iteration {}: Best Cost = {}, parameter = {}, stock = {}".format(it, bestcost[it], bestsol['position'], bestsol['stock']))
        
    # Output
    out = {}
    #out['ants'] = ants
    out['bestsol'] = bestsol
    out['bestcost'] = bestcost
        
    return out
    

def calculate_probability(x,y,phero,alpha,beta):
    #x last variable to be calculated
    #y last day to be calculated
    x=int(x)
    y=int(y)
    denominator=0
    for i in range(varmax):
        denominator+=phero[(x*varmax)+y][((x+1)*varmax)+y+i]
    probability = phero[(x*varmax)+y][(((x+1)*varmax)-1):(((x+1)*varmax)+varmax-1)]/denominator

def next_value(x,y,phero,alpha,beta,n):
    #x last variable to be calculated
    #y last day to be calculated
    x=int(x)
    y=int(y)
    denominator=0
    for i in range(varmax):
        denominator+=phero[(x*varmax)+y][((x+1)*varmax)+i]
    probability = phero[(x*varmax)+y][(((x+1)*varmax)):(((x+1)*varmax)+varmax)]/denominator
    c=np.cumsum(probability)
    r=np.random.rand()
    ind = np.argwhere(r<=c)
    return ind[0][0]




