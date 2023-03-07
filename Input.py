import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import datetime
import ast
# Function which returns subset or r length from n
from itertools import product

get_ipython().run_line_magic('run', 'ACO_algorithm_pure.ipynb')
get_ipython().run_line_magic('run', 'GA_algorithm_pure.ipynb')
get_ipython().run_line_magic('run', 'PSO_algorithm_pure.ipynb')
get_ipython().run_line_magic('run', 'SA_algorithm_pure.ipynb')
get_ipython().run_line_magic('run', 'TS_algorithm_pure.ipynb')


# Model Construction

## Costfunction


#the costfunction is called by the algorithm and the given array(x) is the solution it generated. 
#it consists of n(=number of parts in the change context) solution triplets (y1,d2,d3)
#y1 defines the % reduction of the last devlivery
#d2 defines the last production day (einsatztag)
#d3 defines the last delivery day
#the cost calculation is repeated for each part in a change context
#a solution for 3 parts for instance has 3 triplets, and the array 
#is defined as [p1y1,p1d2,p1d3,p2y1,p2d2,p2d3,p3y1,p3d2,p3d3]
def costfunction(x):
    
    totalcost = 0
    
    n=0
    for i in range(0,parts):
        y1 = int((x[n+i])) # value from algorithm for % of last delivery
        d2 = int((x[n+i+1])) #index from algorithm for possible date
        d3 = int((x[n+i+2])) #index from algorithm for possible date
        y2 = len(np.argwhere(plan_prod_dates[i]<=possible_date_list[d2])) #returns the index of the last entry in prod_plan
        if y2 < 0:
            y2=0
        y3 = len(np.argwhere(call_off_dates[i]<=possible_date_list[d3])) # returns the index of the last entry in call_off
        if y3<0:
            y3=0
        a=np.nonzero(call_off[i][0:y3])
        if y1==100:
            hc=0
        else:
            hc=50
        if len(a[0])==0:
            y1=0
            
        sum_prod = np.sum(plan_prod[i][0:y2])*-1
        sum_stock = np.sum(call_off[i][0:y3],initial=stock[i])
        totalcost = totalcost + const_prod*sum_prod+const_stock[i]*(sum_stock-sum_prod)+hc+(1-(y1/100))*100
        n=n+2
    
    return totalcost

### Supporting Function


#the stock function is called by the algorithm and for the given array(x) the stock levels are calcualated. 
#it consists of n(=number of parts in the change context) solution triplets (y1,d2,d3)
#y1 defines the % reduction of the last devlivery
#d2 defines the last production day (einsatztag)
#d3 defines the last delivery day
#the cost calculation is repeated for each part in a change context
#a solution for 3 parts for instance has 3 triplets, and the array 
#is defined as [p1y1,p1d2,p1d3,p2y1,p2d2,p2d3,p3y1,p3d2,p3d3]
def calc_stock(x): #calculate available stock
    c_stock = [None]*parts
    n=0
    for i in range(0,parts):
        y1 = int((x[n+i]))
        d2 = int((x[n+i+1])) #index from algorithm for possible date
        d3 = int((x[n+i+2])) #index from algorithm for possible date
        y2 = len(np.argwhere(plan_prod_dates[i]<=possible_date_list[d2]))-1 #returns the index of the last entry in prod_plan
        if y2<0:
            y2=0
        y3 = len(np.argwhere(call_off_dates[i]<=possible_date_list[d3]))-1 # returns the index of the last entry in call_off
        if y3<0:
            y3=0
        a=np.nonzero(call_off[i][0:y3])
        if len(a[0])>0:
            save_value=call_off[i][np.max(np.nonzero(call_off[i][0:y3]))]
            store_i=i
            store_max=np.max(np.nonzero(call_off[i][0:y3]))
            call_off[i][np.max(np.nonzero(call_off[i][0:y3]))]=np.around(call_off[i][np.max(np.nonzero(call_off[i][0:y3]))]*(y1/100))
            c_stock[i] = (np.sum(call_off[i][0:y3],initial=stock[i])- np.sum(plan_prod[i][0:y2])*-1)
        else:
            c_stock[i] = stock[i]- np.sum(plan_prod[i][0:y2])*-1
            
        if len(a[0])>0:
            call_off[store_i][store_max]=save_value
        
        n=n+2
        
    current_stock=c_stock
    return current_stock


# Parameter Sets

ga_params = {}
ga_maxit = 1000 # number of iterations of algorithm
ga_npop = 50 # number of population
ga_beta = 1
ga_pc = 1 # child to parent ratio
ga_gamma = 0.9 # adds level of exploration
ga_mu = 0.5 # percentage of genes mutated on average
ga_sigma = 15

ga_params = {'maxit':ga_maxit, 'npop':ga_npop, 'pc':ga_pc, 'gamma':ga_gamma, 'mu': ga_mu, 'sigma': ga_sigma, 'beta':ga_beta}


aco_params = {}
aco_maxit = 1000 # number of iterations of algorithm
aco_npop = 50 # number of ants
aco_alpha = 1
aco_beta = 1
aco_rho = 0.005

aco_params = {'maxit':aco_maxit, 'npop':aco_npop, 'alpha':aco_alpha, 'rho':aco_rho, 'beta':aco_beta}


pso_params = {}
pso_maxit = 1001 # number of iterations of algorithm
pso_npop = 50 # number of particles
pso_rho1 = 0.8
pso_rho2 = 0.3

pso_params = {'maxit':pso_maxit, 'npop':pso_npop, 'rho1':pso_rho1, 'rho2':pso_rho2}


sa_params = {}
sa_maxit = 1000 # number of iterations of algorithm
sa_temp = 2000 # initial temprature
sa_alpha = 0.99999
sa_temp_term = 10
sa_sigma = 15 # value to create searchspace

sa_params = {'maxit':sa_maxit, 'temp':sa_temp, 'alpha':sa_alpha, 'temp_term':sa_temp_term, 'sigma': sa_sigma}


ts_params = {}
ts_maxit = 1000 # number of iterations of algorithm
ts_npop = 20 # size of neigborhood
ts_sigma = 15 # value to create searchspace

ts_params = {'maxit':ts_maxit, 'npop':ts_npop, 'sigma': ts_sigma}


# Load Problem Input from data tables


#opens an empty problem as a dictionary
problem = {}

#input file -> would be the data table 
file_name='Example_testset.xlsx'

#open input file
test_set=pd.read_excel(io=file_name, engine='openpyxl')

#list of unique change batches for which the algorithm is calculating solutions (20 was the number of the initial test set)
batch_list=list(test_set['Batch'][0:20])

#for each change batch, do: 
for EC in range(0,20):
    
    #this takes the change batch as an input -> would be the data in additional tables
    problem_df=pd.read_excel(io=file_name,sheet_name=batch_{}.format(EC), engine='openpyxl')
    
    #number of rows = number of parts / scenarios to be calculated
    parts = len(problem_df['part_old'].dropna())
    
    #number of variables within the solution. currently three determination variables for each part
    detvar = 3
    nvar = detvar*parts
    
    #minimum value of the variables
    #maximum value of the variables (101 as ranges are non-inclusive with python)
    varmin = 0
    varmax = 101
    
    
    #check if all drawings are the same or not. 
    #if all drawings are the same, each part can be introduced seperatley
    no_draw = problem_df['drawing_number'].nunique()
    if no_draw !=1:
        independency = 0
    else:
        independency = 1
        
    #generate solutions space - once operational should be today! Testruns where started on April 25th
    base = datetime.datetime.strptime('25/04/22', '%d/%m/%y')
    
    #generate possible date list for solution space
    possible_date_list = np.array([base + datetime.timedelta(days=x) for x in range(varmax)], dtype='datetime64[ns]')
    
    #generate list of stock of length parts with n empty elements
    stock=[None]*parts
    
    #same for call_offs
    call_off=[None]*parts
    
    #production figures
    plan_prod=[None]*parts
    
    #product related costs
    const_stock=[None]*parts
    
    #call_off dates
    call_off_dates=[None]*parts
    
    #planned production dates
    plan_prod_dates=[None]*parts
    
    #delay penalty
    pen_deliver_mod=[None]*parts
    
    #for each part
    for prt in range(parts):
        
        #-> divide by 1000 due to decimal system, might not be necessary
        stock[prt]=int(problem_df['Current_Stocklevel'][prt]/1000)
        
        #Take array of call_offs_list -> directly in impact, not necessary to translate ast.liter
        call_off[prt]=ast.literal_eval(problem_df['Call_offs_List'][prt])
        
        #take array of delivery dates - transformations in impact probably not necessary
        call_off_dates[prt]=(pd.to_datetime(problem_df['Delivery_Date_List'][prt][2:-2].split(','),yearfirst=True)).to_numpy()
        
        #take array of production plan
        plan_prod[prt]=ast.literal_eval(problem_df['Consumption'][prt])
        
        #take array of consumation date
        plan_prod_dates[prt]=(pd.to_datetime(problem_df['Consumation_Date_List'][prt][2:-2].split(','),yearfirst=True)).to_numpy()
        
        #cost price for each part
        const_stock[prt]=problem_df['unit_price'][prt]
        
        #delay penalty for each part
        pen_deliver_mod[prt]=problem_df['unit_price'][prt]/10
        
        #cost price for each change context -> taken from level 0
        const_prod=test_set['warranty'][EC]
           
            
    problem = {'costfunc':costfunction, 'nvar':nvar, 'varmin':varmin, 'varmax':varmax, 'calc_stock': calc_stock, 'parts':parts, 'independency':independency}
    
    # base date was given as the 25th of April, as this was the earliest  starting date of the deliveries in the test cases
    base = datetime.datetime.strptime('25/04/22', '%d/%m/%y')
    possible_date_list = np.array([base + datetime.timedelta(days=x) for x in range(varmax+1)], dtype='datetime64[ns]')

    algorithm_list=['aco','ga','pso','ts','sa']
    trackrecord[EC]=[None]*len(algorithm_list)
    timerecord[EC]=[None]*len(algorithm_list)
    trackrecord_best[EC]=[None]*len(algorithm_list)
    trackrecord_best_sol[EC]=[None]*len(algorithm_list)
    for alg in range(len(algorithm_list)):
        trackrecord[EC][alg]=[]
        timerecord[EC][alg]=[]
        trackrecord_best[EC][alg]=[]
        trackrecord_best_sol[EC][alg]=[]
    
    for run in range (200):
        print(run)
        # Import predefined initial solutions
        fileimport= "Init_sol/init_pos_pop_{}_{}.xlsx".format(EC, run)
        init_pop_df=pd.read_excel(fileimport, header=None, engine='openpyxl')
        init_pop=[]
        init_cost=np.inf
        for n in range(50):
            init_pos=init_pop_df.iloc[n].to_numpy()
            temp_cost=costfunction(init_pos)
            if temp_cost<init_cost:
                best_init=init_pos
            init_pop.append(init_pos)

        #run aco
        print('aco started')
        start = datetime.datetime.now()
        aco_out = aco_run(problem,aco_params)
        end = datetime.datetime.now()
        print((end-start).total_seconds())
        timerecord[EC][0].append((end-start).total_seconds())
        trackrecord[EC][0].append(aco_out['bestcost'])
        trackrecord_best[EC][0].append(min(aco_out['bestcost']))
        trackrecord_best_sol[EC][0].append(aco_out['bestsol'])

        #run ga
        print('ga started')
        start = datetime.datetime.now()
        ga_out = ga_run(problem,ga_params)
        end = datetime.datetime.now()
        print((end-start).total_seconds())
        timerecord[EC][1].append((end-start).total_seconds())
        trackrecord[EC][1].append(ga_out['bestcost'])
        trackrecord_best[EC][1].append(min(ga_out['bestcost']))
        trackrecord_best_sol[EC][1].append(ga_out['bestsol'])

        #run pso
        print('pso started')
        start = datetime.datetime.now()
        pso_out = pso_run(problem,pso_params)
        end = datetime.datetime.now()
        print((end-start).total_seconds())
        timerecord[EC][2].append((end-start).total_seconds())
        trackrecord[EC][2].append(pso_out['bestcost'])
        trackrecord_best[EC][2].append(min(pso_out['bestcost']))
        trackrecord_best_sol[EC][2].append(pso_out['bestsol'])

        #run ts
        print('ts started')
        start = datetime.datetime.now()
        ts_out = ts_run(problem, ts_params)
        end = datetime.datetime.now()
        print((end-start).total_seconds())
        timerecord[EC][3].append((end-start).total_seconds())
        trackrecord[EC][3].append(ts_out['bestcost'])
        trackrecord_best[EC][3].append(min(ts_out['bestcost']))
        trackrecord_best_sol[EC][3].append(ts_out['bestsol'])

        #run sa
        print('sa started')
        start = datetime.datetime.now()
        sa_out = sa_run(problem, sa_params)
        end = datetime.datetime.now()
        print((end-start).total_seconds())
        timerecord[EC][4].append((end-start).total_seconds())
        trackrecord[EC][4].append(sa_out['bestcost'])
        trackrecord_best[EC][4].append(min(sa_out['bestcost']))
        trackrecord_best_sol[EC][4].append(sa_out['bestsol'])

fileexport_trr='trackrecord.csv'
fileexport_tir='timerecord.csv'
fileexport_bc='best_cost.csv'
fileexport_bs='trackrecord_bestsol.csv'
df_trr_aco=pd.DataFrame(trackrecord).to_csv(fileexport_trr,index=False, header=False)
df_tir_aco=pd.DataFrame(timerecord).to_csv(fileexport_tir,index=False, header=False)
df_bc_aco=pd.DataFrame(trackrecord_best).to_csv(fileexport_bc,index=False, header=False)
df_bs_aco=pd.DataFrame(trackrecord_best_sol).to_csv(fileexport_bs,index=False, header=False)





