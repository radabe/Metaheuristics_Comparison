""" This script defines the Particle Swarm Optimization Algorithm. 

***
Its base functions and code were taken from: Mostapha Kalami Heris, Practical Genetic Algorithms in Python and MATLAB – Video Tutorial (URL: https://yarpiz.com/632/ypga191215-practical-genetic-algorithms-in-python-and-matlab), Yarpiz, 2020. 
and adapted for the Particle Swarm Optimization application and to the Engineering Change domain.
***

"""

# The PSO Algorithm

def pso_run(problem, params):
    
    #Problem Information
    cf = problem['costfunc']
    st = problem['calc_stock']
    number_var = problem['nvar']
    init_pop = problem['init_pop']
    vmin = problem['varmin']
    vmax = problem['varmax']+1
    no_parts = problem['parts']
    indep = problem['independency']
    validity=0
    
    # Parameters
    maxit = params['maxit'] # number of iterations
    rho1 = params['rho1'] # Evaporation rate [0.01, 0.2]
    rho2 = params['rho2'] # Evaporation rate [0.01, 0.2]
    kappa = params['npop'] #Number of ants [10,50]
    
    # Empty Individual Template
    empty_individual = {}
    empty_individual = {'position':None,'cost':None, 'stock':None}
    empty_individual['position']=init_pos

    swarmbest=copy.deepcopy(empty_individual)
    swarmbest['cost']=np.inf
    particlebest=[None]*kappa
    for n in range(0,kappa):
        particlebest[n] = copy.deepcopy(empty_individual)
        particlebest[n]['cost']=np.inf

    velocity=[]
    velocity.append([None]*kappa)
    swarm=list(range(maxit))
    particle = [None]*kappa
    
    for n in range(kappa):
        velocity[0][n]=np.zeros((1,number_var))
        
    # Best Cost of Iterations
    bestcost = np.empty(maxit-1)

    #initialise swarm
    particle = [None]*kappa
    for n in range (0, kappa):
        validity=0
        while validity < 1:
            
        particle[n]= copy.deepcopy(empty_individual)
        particle[n]['position']=init_pop[n]
            #toggle on off for effectivity date on single day
            if indep ==0:
                for j in range(number_var):
                    if ((j%3==1)&(j>2)):
                        particle[n]['position'][j]=particle[n]['position'][j-int(number_var/no_parts)]
        particle[n]['stock']=st(particle[n]['position'])
        particle[n]['cost']=cf(particle[n]['position'])
    
            validity=0
            for m in range (0, no_parts):
                if particle[n]['stock'][m]<0:
                    validity -= 1
                else:     
                    validity += 1
            validity/= no_parts
            
        if particle[n]['cost']<swarmbest['cost']:
            swarmbest=copy.deepcopy(particle[n])
        if particle[n]['cost']<particlebest[n]['cost']:
            particlebest[n]=copy.deepcopy(particle[n])

    swarm[0]= particle
    temp_velocity=[None]*kappa
    #main loop
    for it in range (0,maxit-1): 

        for n in range(kappa):
            validity=0
            temp_velocity[n]=0.9*(velocity[it][n])+rho1*np.random.rand()*(particlebest[n]['position']-swarm[it][n]['position'])+rho2*np.random.rand()*(swarmbest['position']-swarm[it][n]['position'])
            temp_velocity[n]=apply_bound_velo(temp_velocity[n],-5,5)
            particle[n]['position']=np.around(particle[n]['position'][0]+temp_velocity[n][0])
            apply_bound(particle[n], vmin, vmax)
            particle[n]['stock']=st(particle[n]['position'])
            
            for m in range (0, no_parts):
                
                if particle[n]['stock'][m]<0:
                    validity -= 1
                else:     
                    validity += 1 
            validity/= no_parts
            if validity==1:
                particle[n]['cost']=cf(particle[n]['position'])
                if particle[n]['cost']<=particlebest[n]['cost']:
                    particlebest[n]=copy.deepcopy(particle[n])
                if particle[n]['cost']<=swarmbest['cost']:
                    swarmbest=copy.deepcopy(particle[n])

        velocity.append(temp_velocity)
        swarm[it+1]=particle   
        
        # Store Best Cost
        bestcost[it]=swarmbest['cost']
        # Show Iteration Information
        print("Iteration {}: Best Cost = {}, parameter = {}, stock = {}".format(it, bestcost[it], swarmbest['position'], swarmbest['stock']))
        
    # Output
    out = {}
    out['swarm'] = swarm
    out['bestsol'] = swarmbest
    out['bestcost'] = bestcost
        
    return out

def apply_bound (x, vmin, vmax):
    x['position'] = np.maximum(x['position'], vmin) #replace invalid value with minimum possible value
    x['position'] = np.minimum(x['position'], vmax-1) #replace invalid value with maximum possibe value
    return x


def apply_bound_velo (x, a, b):
    x = np.maximum(x, a) #replace invalid value with minimum possible value
    x = np.minimum(x, b) #replace invalid value with maximum possibe value
    return x

