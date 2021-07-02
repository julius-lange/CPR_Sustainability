#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 19:13:36 2021

@author: juliuslange
"""

import networkx as nx
import numpy as np
import random
import matplotlib.colors
import mean_field as mf
from scipy.stats import ttest_ind_from_stats

class player:
    '''
    A class for player Oobjects which sit at the nodes of the network
     
    ...
    
    Attributes
    __________
    
    index : int 
        An index corresponding to the node number in the network
    strategy : int
        Player strategy, 0 is defection, 1 cooperation
    payoff : float
        Single turn payoff
    total_strat : int
        Sum of the values the strat variable take over the lifetime
    total_payoff : float
        Sum of all of the payoffs over the lifetime
    total_payoff_squares : float
        Sum of the squares for clculating uncertainties
    neighbour_array : np.array
        Array of the indices of a players' neighbours
    N : inr
        Number of nodes in the network
    
    Methods
    _______
    
    harvest(R,e_C, e_D, w)
        Calculate the individual payoffs of a ll players and return the total
        harvest
    update1-6(various)
        Updating functions to update one node according to one specific 
        procedure
    update(ntype,G,*argv)
        Generalised updating function taking the number of the corresponding 
        procedure as argument with a given network G
        
    '''
    
    index=-1
    strat=0
    payoff=0
    total_strat=0
    total_payoff=0
    total_payoff_squares=0
    neighbour_array=np.array([])
    N=0
    
    def __init__(self,strat,G,i):
        '''
        Parameters
        __________
        
        strat : int
            Strategy of node
        G : networkx.network()
            Network
        i : int
            Index
        '''
        self.index=i
        self.strategy=strat
        self.payoff=0
        self.total_strat=strat
        self.total_payoff=0
        self.total_payoff_squares=0
        self.neighbour_array=neighbours(G,i)
        self.N=G.number_of_nodes()

    
    def harvest(self,R,e_C, e_D, w):
        ''' Function to update individual payoff value according to strategy
        
        Update individual payoff (and cumulative and square payoff)
        according to startegy and general payoff function
        
        Parameters
        __________
        R : float
            Resource level
        e_C : float
            Harvesting effort for Cooperators
        e_D : float
            Harvesting effort for Defectors
        w : float
            marginal cost of effort
           
        Returns
        _______
        harvest : float
            Individual harvest
    
        '''
        harvest=0
        if self.strategy==1:
            harvest=e_C/self.N*(R)
            #print(self.index,', C harvested')
            self.payoff=e_C/self.N*(R-w)
        elif self.strategy==0:
            harvest=e_D/self.N*(R)
            #print(self.index,',D harvested')
            self.payoff=e_D/self.N*(R-w)

           
        else: print("invalid strategy")
        
        #P=harvest
        #self.payoff=harvest
        self.total_strat+=self.strategy
        self.total_payoff_squares+=self.payoff*self.payoff
        self.total_payoff+=self.payoff
        
        return harvest
    
    def update1(self,G):
        '''RHPM Updating Procedure
        
        Function to mirror the strategy of the neighbour with the highest 
        payoff
        
        Parameters
        __________
        
        G : networkx.network
            Network
        
        '''
        total_neighbour_profit=0
        #print(self.neighbour_array)
        for i in range(self.neighbour_array.size):
            total_neighbour_profit+=G.nodes[self.neighbour_array[i]]["Player"].payoff
            #print(i,G.nodes[i]["Player"].payoff)
        average_neighbour_profit=total_neighbour_profit/self.neighbour_array.size
        #print(self.payoff-average_neighbour_profit)
        if cooperators(G)==G.number_of_nodes() or cooperators(G)==0:
            return
        if self.payoff<average_neighbour_profit:
            if self.strategy==0:
                self.strategy=1
                #print('change D->C, node:', self.index)
            elif self.strategy==1:
                self.strategy=0
                #print('change C->D, node:', self.index)

            else: print('ERROR')
        return
    
    def update2(self,G):
        '''IHPM - Irrational Highest Profit Mirroring Updating Procedure
        
        Function to mirror the strategy of the neighbour with the highest 
        payoff. If the individual payoff is equal to the average neighbour 
        profit, a change in startegy occurs with probability p
        
        Parameters
        __________
        
        G : networkx.network
            Network
        p : float
            Probability of change when the individual payoff equals the 
            average communal payoff
        
        Raises
        ______
        ERROR : string
            Error message if self.strategy is not 0 or 1
        
        '''
        total_neighbour_profit=0
        #print(self.neighbour_array)
        for i in range(self.neighbour_array.size):
            total_neighbour_profit+=G.nodes[self.neighbour_array[i]]["Player"].payoff
            #print(i,G.nodes[i]["Player"].payoff)
        average_neighbour_profit=total_neighbour_profit/self.neighbour_array.size
        #print(self.payoff, average_neighbour_profit)
        if self.payoff<average_neighbour_profit:
            if self.strategy==0:
                self.strategy=1
                #print('change D->C, node:', self.index)
            elif self.strategy==1:
                self.strategy=0
                #print('change C->D, node:', self.index)

            else: print('ERROR')
        
        elif self.payoff==average_neighbour_profit:
            guess=random.random()
            p=1
            if guess<p:
                if self.strategy==0:
                    self.strategy=1
                    #print('change D->C, node:', self.index)
                elif self.strategy==1:
                    self.strategy=0
                    #print('change C->D, node:', self.index)  
        
        return
    
    def update3(self,G,R,w,e_C,e_D):
        '''GP - Global Prediction Updating Procedure
        
        Function to update the strategy of one player.
            1. Calculate resource level next turn
            2. Choose strategy which will be best (dependent on R and w)
        
        Parameters
        __________
        
        G : networkx.network
            Network
        R : float
            Resource level
        w : float
            Marginal cost of effort
        e_C : float
            Harvesting effort for Cooperators
        e_D : float
            Harvesting effort for Defectors
        
        
        '''
        x=cooperators(G)/G.number_of_nodes()
        R_prospective=R+R*(1-R)-R*(e_C*x+e_D*(1-x))
        if R_prospective<w:
            self.strategy=1
        else:
            self.strategy=0
        return
    
    def update4(self,G,R,w,e_C,e_D): 
        '''LP - Local Prediction Updating Procedure
        
        Function to update the strategy of one player.
            1. Calculate resource level next turn according to strategy 
               profile of player neighbours
            2. Choose strategy which will be best (dependent on R and w)
        
        Parameters
        __________
        
        G : networkx.network
            Network
        R : float
            Resource level
        w : float
            Marginal cost of effort
        e_C : float
            Harvesting effort for Cooperators
        e_D : float
            Harvesting effort for Defectors
        
        '''
        strategies=[]
        for i in range(self.neighbour_array.size):
            strategies.append(G.nodes[self.neighbour_array[i]]["Player"].strategy)

        x=sum(strategies)/self.neighbour_array.size
        #print(strategies)
        #print(x)
        R_prospective=R+R*(1-R)-R*(e_C*x+e_D*(1-x))
        if R_prospective<w:
            self.strategy=1
        else:
            self.strategy=0
        return
    
    def update5(self,G,R,w,e_C,e_D):
        '''Historical Profit Updating Procedure
        
        Function to update the strategy of one player.
            1. Calculate the total cumulative payoffs of all neighbours
            2. Choose strategy of neighbour with the highest cumulative payoff
        
        Parameters
        __________
        
        G : networkx.network
            Network
        R : float
            Resource level
        w : float
            Marginal cost of effort
        e_C : float
            Harvesting effort for Cooperators
        e_D : float
            Harvesting effort for Defectors
        
        '''
        total_payoffs=[]
        strategies=[]
        for i in range(self.neighbour_array.size):
            total_payoffs.append(G.nodes[self.neighbour_array[i]]["Player"].total_payoff)
            strategies.append(G.nodes[self.neighbour_array[i]]["Player"].strategy)
        #print(self.neighbour_array[i])
        total_payoffs=np.array(total_payoffs)
        #print(total_payoffs)
        #print(strategies)
        
        index_max=np.argmax(total_payoffs)
        #print("best index: ", index_max)
        #print(self.strategy)
        self.strategy=strategies[index_max]
        #print(self.strategy)

        return
    
    def update6(self,G):
        '''PS - Proportional Selection Updating Procedure
        
        Function to update the strategy of one player.
            1. Calculate the payoff sums for defectors and cooperators among 
               neighbours 
            2. Choose strategy proportional to the payoff sums
        
        Parameters
        __________
        
        G : networkx.network
            Network
       
        '''
        payoffs=[]
        strategies=[]
        for i in range(self.neighbour_array.size):
            payoffs.append(G.nodes[self.neighbour_array[i]]["Player"].payoff)
            strategies.append(G.nodes[self.neighbour_array[i]]["Player"].strategy)
        C_payoffs=0
        D_payoffs=0
        for i in range(self.neighbour_array.size):
            if strategies[i]==0:
               C_payoffs+=payoffs[i]
            else:
                D_payoffs+=payoffs[i]
                
        total_payoff=C_payoffs+D_payoffs
        
        guess=random.random()
        if guess<C_payoffs/total_payoff:
            self.strategy=0
        else: 
            self.strategy=1
            
        return
        
    def update(self,ntype,G,*argv):
        '''General Updating Procedure
        
        Update chosen node according to procedure determined by ntype
        
        Parameters
        __________
        
        ntype : int
            Type of updating procedure:
                1: RHPM
                2: IHPM
                3: GP
                4: LP
                5: HP
                6: PS
        G : networkx.network
            Network
        
        '''
        if ntype==1: #Rational Profit Maximisation
            player.update1(self,G)
        elif ntype==2: #Irrational Profit Maximisation
            #p=argv[0]
            player.update2(self,G)
        elif ntype==3: #Global Prediction Profit Maximisation
            R=argv[0]
            w=argv[1]
            e_C=argv[2]
            e_D=argv[3]
            player.update3(self,G,R,w,e_C,e_D)
        elif ntype==4: #Local Prediction Profit Maximisation
            R=argv[0]
            w=argv[1]
            e_C=argv[2]
            e_D=argv[3]
            player.update4(self,G,R,w,e_C,e_D)
        elif ntype==5: #Average Local Total Payoff
            R=argv[0]
            w=argv[1]
            e_C=argv[2]
            e_D=argv[3]
            player.update5(self,G,R,w,e_C,e_D)
        elif ntype==6: #proportional Selection
            R=argv[0]
            player.update6(self,G)
        else: return
    
    
def neighbours(G,n):
    '''Initiating Neighbour Array for Player n
    
    Function to fill neighbour array of player n
    
    Parameters
    __________
    
    G : networkx.network
        Network
    n: int 
        Node index of node to initiate
        
    Returns
    _______
    
    neighbours : np.array
        Array of indices connected to n
    '''
    A=nx.all_neighbors(G,n)
    neighbours=np.array([])
    while True:
        try:
            item = next(A)
            neighbours=np.append(neighbours,item)    
        except StopIteration:
            return neighbours
        
def cooperators(G):
    '''Calcuating Number of Cooperators in a Network
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    C : int
        Number of cooperators
    
    '''
    C=0
    #print(type(G))
    N=G.number_of_nodes()
    for i in range(N):
        C+=G.nodes[i]["Player"].strategy
        
    return C

def strategy_array(G):
    '''Return Array of Strategies Corresponding to Players 0 to n
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    strat_array : np.array
       Array of strategies corresponding to all nodes
    
    '''
    strat_array=np.array([])
    for i in range(G.number_of_nodes()):
        strat_array=np.append(strat_array,G.nodes[i]["Player"].strategy)
    return strat_array

def degree_strategy_array(G):
    '''Return Degree of Average Strategy of Players of Degree k
    
    Calculate strategy array, then calculate average strategy for nodes of 
    degree 0 to k_max
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    unique_degree : np.array
        Array of degrees represented in the network
    degree_frequency : np.array
        Frequency of cooperators of all degrees of unique_degree[i]
    degree_frequency_error : np.array
        Array of standard deviations associated with degree_frequency
    degree_count : np.array
        Number of nodes with degree unique_degree[i]
    '''
    
    N=G.number_of_nodes()
    degree_array=np.array(G.degree())[:,1]
    unique_degree=np.unique(degree_array)
    
    degree_frequency=np.array([])
    degree_frequency_error=np.array([])
    degree_count=np.array([])

    for i in range(unique_degree.size):
        counter=0
        strat_sum=0
        for j in range(N):
            if G.degree(j)==unique_degree[i]:
                counter+=1
                strat_sum+=G.nodes[j]["Player"].strategy
        mean=strat_sum/counter
        if mean==0 or mean==1: error=0
        else: error=np.sqrt(abs(mean-mean*mean))
        degree_frequency=np.append(degree_frequency,mean)
        degree_frequency_error=np.append(degree_frequency_error,error)
        degree_count=np.append(degree_count,counter)
    return unique_degree,degree_frequency,degree_frequency_error,degree_count

def degree_sequence(G):
    '''Returns Degree Sequency of Network G
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    deg_seq : np.array
        Array of the degrees of node number i
    '''
    deg_seq=[]
    for i in G.nodes:
        deg_seq.append(G.nodes[i]["Player"].neighbour_array.size)
    deg_seq=np.array(deg_seq)
    return deg_seq

def average_strategy_array(G,T):
    '''Returns Array of Average Strategies
    
    COMPARE: 
        strategy_array(G), just consider total_strat and divide by T
    
    '''
    average_strategy=np.array([])
    average_strategy_error=np.array([])

    for i in G.nodes:
        mean=G.nodes[i]["Player"].total_strat/(T+1)
        if mean==0 or mean==1: error=0
        else: error=np.sqrt(abs(mean-mean*mean))
        average_strategy=np.append(average_strategy,mean)
        average_strategy_error=np.append(average_strategy_error,error)
    return average_strategy,average_strategy_error

def average_degree_strategy_array(G,T):
    '''Returns Degree Sorted Array of Average Strategies
    
    COMPARE: 
        degree_strategy_array(G), just consider total_strat and divide by T
    
    '''
    
    N=G.number_of_nodes()
    degree_array=np.array(G.degree())[:,1]
    unique_degree=np.unique(degree_array)
    
    degree_frequency=np.array([])
    degree_frequency_error=np.array([])
    degree_count=np.array([])
    
    average_strategies=average_strategy_array(G,T)[0]

    for i in range(unique_degree.size):
        counter=0
        strat_sum=0
        strat_squares=0
        for j in range(N):
            if G.degree(j)==unique_degree[i]:
                counter+=1
                strat_sum+=average_strategies[j]
                strat_squares+=average_strategies[j]*average_strategies[j]
        mean=strat_sum/counter
        mean_squares=strat_squares/counter
        if mean==0 or mean==1: error=0
        else: error=np.sqrt(abs(mean_squares-mean*mean))
        degree_frequency=np.append(degree_frequency,mean)
        degree_frequency_error=np.append(degree_frequency_error,error)
        degree_count=np.append(degree_count,counter)
    return unique_degree,degree_frequency,degree_frequency_error,degree_count

def initialise(G,C_i):
    '''Initialise Network G with C_i Cooperators
    
    Parameters
    __________
    
    G : networkx.network
        Network
    C_i : int
        Number of initial cooperators
        
    Returns
    _______
    
    G : networkx.network
        Network
    '''
    nodes_list=[]
    for i in range(G.number_of_nodes()):
        nodes_list.append(i)
    selection=random.sample(nodes_list,C_i )
    for i in G.nodes:
        if i in selection:
            G.nodes[i]["Player"]=player(1,G,i)
        else:
            G.nodes[i]["Player"]=player(0,G,i)
    return G

def resource_change(R,G,e_C,e_D):
    '''Calculate Resource Change for one Time-Step
    
    Resource change is calculated by the sum of the logistic growth component 
    and the harvest of all players
    
    Parameters
    __________
    
    R : float
        Resource level
    G : networkx.network
        Network
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
        
    Returns
    _______
    
    dR : float
        Resource Change
    
    '''
    x=cooperators(G)/G.number_of_nodes()
    #print(R*(x*e_C+(1-x)*e_D))
    dR=R*(1-R)-R*(x*e_C+(1-x)*e_D)
    return dR

def timestep(G,R,e_C,e_D,w,n,ntype):
    '''Function Propagating the System for One Time Step
    
    Function selecting n nodes to be updated, updating them, and changing 
    resource levels
    
    Parameters
    __________
    
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
        
    Returns
    _______
    
    G : networkx.network
        Network
    R : float
        Resource level
    '''
    N=G.number_of_nodes()
    #nodes_list=list(np.arange(N))
    nodes_list=np.arange(N).tolist()
    z=0
    for i in range(N):
        #G.nodes[i]["Player"].harvest(R,e_C, e_D, w)
        z+=G.nodes[i]["Player"].harvest(R,e_C, e_D, w)
    
    indices=random.sample(nodes_list,n )
    #index=random.randint(0,N-1)
    for i in range(n):
        G.nodes[indices[i]]["Player"].update(ntype,G,R,w,e_C,e_D)
    #print(index,G.nodes[index]["Player"].neighbour_array)
    
    '''
    G.nodes[Z]["Player"].update1(G)
    x=cooperators(G)/G.number_of_nodes()
    '''
    #print(cooperators(G))
    #print(R*(x*e_C+(1-x)*e_D))
    #print("here:",z)
    #print(R*(1-R))
    #print(resource_change(R,G,e_C,e_D))
    #print(R*(1-R)-z)
    R=R+R*(1-R)-z
    return G,R

def propagate(T,G,R_i,e_C,e_D,w,n,ntype):
    '''Propagate the System for T time-steps
    
    Propagate the system for T time-steps by executing "timestep" T times and 
    returning array of time, resource level,and cooperator fraction
    
    Parameters
    __________
        
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
    
    Returns
    _______
    
    t_array : np.array
        Array of time values
    r_array : np.array
        Resource level at time t
    x_array : np.array
        Cooperator fraction at time t
    
    '''
    t_array=np.array([])
    r_array=np.array([])
    x_array=np.array([])
    
    R=R_i
    for i in range(T):  
        #print(i)
        t_array=np.append(t_array,i)
        r_array=np.append(r_array,R)
        x_array=np.append(x_array,cooperators(G))
        G,R=timestep(G,R,e_C,e_D,w,n,ntype)
        
    return t_array,r_array,x_array

def print_network(G):
    '''Print Simple Network Variables
    
    Prints number of nodes, number of edges, whether network is initialised,
     - index - strategy - payoff - (for all nodes), mean strategy, 
    degree strategy array
     
    Parameters
    __________
    
    G : networkx.network
        Network
    '''
    #print('Resource: ', R)
    print('Nodes: ', G.number_of_nodes())
    print('Edges: ', G.size())
    if G.nodes[0]["Player"]:
        v=0
        print('Network initialised')
        for i in G.nodes:
            print(G.nodes[i]["Player"].index,G.nodes[i]["Player"].strategy,G.nodes[i]["Player"].payoff)
            v+=G.nodes[i]["Player"].payoff
        print('Mean Strategy:',np.mean(strategy_array(G)))
        print('Degree Strategy Array:',degree_strategy_array(G))
    return
    
def steady_state(T,G,R_i,e_C,e_D,w,n,ntype,T_cut,*argv):
    '''Propagate Network for T Time-Steps and Return Steady State Variables
    
    Parameters
    __________
    
    T : int
        Time after which to stop propagation
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
    T_cut : int
        Number after which to start averaging for steady state data
        
    Returns
    _______
    
    mean_r : float
        Average steady state resource level
    std_r : float
        Standard deviation of steady state resource level
    mean_x : float
        Average steady state cooperator fraction
    std_x : float
        Standard deviation of steady state cooperator fraction
    
    '''
    t_array,r_array,x_array=propagate(T,G,R_i,e_C,e_D,w,n,ntype,*argv)
    r2_array=r_array[T_cut:]
    x2_array=x_array[T_cut:]
    #print(r_array)
    #print(r2_array)
    mean_r=np.mean(r2_array)
    std_r=np.std(r2_array)
    mean_x=np.mean(x2_array)
    std_x=np.std(x2_array)
    
    #higher_array=(r_array<mean_r+3*std_r)
    #lower_array=(r_array>mean_r-3*std_r)
    #t=0
    
    #print("higher array", higher_array)
    #print("lower array", lower_array)

    
    #if higher_array.size<2: t=0
    #elif lower_array.size<2: t=0
    #else:
       # t=np.where(higher_array==lower_array)[0][1]
    
    return mean_r,std_r,mean_x,std_x

def time_to_extinction(G,R_i,e_C,e_D,n,ntype,*argv):
    '''Calculate Time it Takes for the Resource to Become Extinct
    
    Propagate the system until the resource level drops below R=0.001
    
    Parameters
    __________
    

    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
    
        
    Returns
    _______
    
    ttext : int
        Time to extinction
    t_array,r_array,x_array : np.array
        Compare "propagate"
    '''
    w=0
    t_array=np.array([])
    r_array=np.array([])
    x_array=np.array([])
    R_ext=0.001
    #T=5*G.number_of_nodes()
    R=R_i
    #ttext=0
    #while(R>R_ext):
       # G,R=timestep(G,R,e_C,e_D,w,n,ntype,*argv)
       # ttext+=1
    ttext=0   
    for i in range(1000):
        t_array=np.append(t_array,i)
        r_array=np.append(r_array,R)
        x_array=np.append(x_array,cooperators(G))
        G,R=timestep(G,R,e_C,e_D,w,n,ntype,*argv)
        if R<R_ext:
            ttext=i
            break
    #t_array,r_array,x_array=propagate(T,G,R_i,e_C,e_D,w,n,ntype)
    #ttext=np.where(r_array<R_ext)[0][0]
    return ttext,t_array,r_array,x_array

def degree_propagation(T,G,R_i,e_C,e_D,w,n,ntype,*argv):
    t_array=np.array([])
    r_array=np.array([])
    x_array=np.array([])
    R=R_i
    unique_degrees=degree_strategy_array(G)[0]
    for i in range(T):  
        #print(i)
        t_array=np.append(t_array,i)
        r_array=np.append(r_array,R)
        #G_copy=G
        degree_frequency=degree_strategy_array(G)[1]

        x_array=np.append(x_array,degree_frequency)
        G,R=timestep(G,R,e_C,e_D,w,n,ntype,*argv)
    x_array.shape=(T,unique_degrees.size) 
          
    return t_array,r_array,x_array

def total_payoff(T,G,R_i,e_C,e_D,w,n,ntype,*argv):
    '''Calculate Total Payoff after time T
    
    Parameters
    __________
    
    T : int
        Time after which to stop propagation
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
        
    Returns
    _______
    
    sum(total_payoff_array) : np.float
        Total cumulative payoff for all nodes
    
    '''
    
    t_array,r_array,x_array=propagate(T,G,R_i,e_C,e_D,w,n,ntype,*argv)
    
    total_payoff_array=np.array([])
    
    for i in G.nodes:
        total_payoff_array=np.append(total_payoff_array,G.nodes[i]["Player"].total_payoff)
    
    return sum(total_payoff_array)

def degree_average_total_payoff(T,G,R_i,e_C,e_D,w,n,ntype,*argv):
    '''Degree Average Cumulative Payoff after time T
    
    Parameters
    __________
    
    T : int
        Time after which to stop propagation
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
        
    Returns
    _______
    
    unique_degrees : np.array
        Unique degrees (compare "degree_strategies")
    final_array : np.array
        Array of the average cumulated payoff for all nodes of degree k
    
    '''
    
    
    total_payoff_array=total_payoff(T,G,R_i,e_C,e_D,w,n,ntype,*argv)
    
    deg_sequence=degree_sequence(G)
    
    degree_payoff_array=np.zeros(np.amax(deg_sequence)+1)
    counter_array=np.zeros(np.amax(deg_sequence)+1)
    
    final_array=np.array([])
    unique_degrees=np.array([])
    
    print(degree_payoff_array.size)
    print(counter_array.size)
    
    for i in range(G.number_of_nodes()):
        payoff=total_payoff_array[i]
        degree=deg_sequence[i]
        #print(degree)
        degree_payoff_array[degree]+=payoff
        counter_array[degree]+=1
    
    print(counter_array)
    
    for i in range(np.amax(deg_sequence)+1):
        if counter_array[i]!=0:
            degree_payoff_array[i]/=counter_array[i]
            final_array=np.append(final_array,degree_payoff_array[i])
            unique_degrees=np.append(unique_degrees,i)
    #degree_payoff_array/=counter_array
    
    return unique_degrees,final_array
    
def draw_current_strat(G):
    '''Draw Network According to Current Strategies
    
    Parameters
    __________
    
    G : networkx.network
        Network
    
    '''
    cmap=matplotlib.cm.seismic
    nx.draw(G,with_labels=True,node_color=cmap(strategy_array(G)))
    return

def draw_avg_strat(G,T):
    '''Draw Network According to Average Strategies after time T
    
    Parameters
    __________
    
    G : networkx.network
        Network
    T : int
        Time through which the network has been propagated
    
    '''
    cmap=matplotlib.cm.seismic
    nx.draw(G,with_labels=True,node_color=cmap(average_strategy_array(G,T)[0]))
    return

def change_events(T,G,R_i,e_C,e_D,w,n,ntype,*argv):
    '''Calculate Frequencies of Calls to "update" which Led to a Strategy 
    Change
    
    
    Parameters
    __________
    
    T : int
        Time after which to stop propagation
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of bpdating procedure
        
    Returns
    _______
    
    t_array, r_array, x_array : np.array
        Compare with "propagate"
    change_array : np.array
        Array which contains a 0 if no change occurs and 1 if a change occured
    '''
    t_array=np.array([])
    r_array=np.array([])
    x_array=np.array([])
    change_array=np.array([])
    
    N=G.number_of_nodes()
    #nodes_list=list(np.arange(N))

    R=R_i
    for i in range(T):  
        #print(i)
        t_array=np.append(t_array,i)
        r_array=np.append(r_array,R)
        x_array=np.append(x_array,cooperators(G))
        #nodes_list=list(np.arange(N))
        z=0
        for i in range(N):
            #G.nodes[i]["Player"].harvest(R,e_C, e_D, w)
            z+=G.nodes[i]["Player"].harvest(R,e_C, e_D, w)
    
        index=random.randint(0,N-1)
        old_strategy=G.nodes[index]["Player"].strategy
        #print(old_strategy)
        
        G.nodes[index]["Player"].update(ntype,G,R,w,e_C,e_D,*argv)
        #print(index,G.nodes[index]["Player"].neighbour_array)
        new_strategy=G.nodes[index]["Player"].strategy
        if new_strategy!=old_strategy:
            change_array=np.append(change_array,1)
        else: change_array=np.append(change_array,0)

        R=R+R*(1-R)-z
        
    return t_array,r_array,x_array,change_array

def change_hist(T,G,R_i,e_C,e_D,w,n,ntype,binsize,*argv):
    '''Calculate Histogram of Number of Change Events in "binsize" Time Steps
    
    Parameters
    __________
    
    T : int
        Time after which to stop propagation
    G : networkx.network
        Network
    R_i : float
        Initial resource level
    e_C : float
        Harvesting effort for Cooperators
    e_D : float
        Harvesting effort for Defectors
    w : float
        Marginal cost of effort
    n : int
        Number of nodes to be updated
    ntype : int
        Type of updating procedure
    binsize : int
        Binwidth for Histogram
        
    Returns
    _______
    
    bin_array : np.array
        Array containing t values spaced by "binsize"
    content_array : np.array
        Array containing the number of change events occuring in each binsize
        interval
    t_copy,r_array : np.array
        Compare "propagate"
    '''
    
    t_array,r_array,x_array,change_array=change_events(T,G,R_i,e_C,e_D,w,n,ntype,*argv)
    
    #remainder=T%binsize
    t_copy=t_array
    
    nbins=int(T/binsize)
    N=nbins*binsize
    #print(nbins,binsize,N)
    
    change_array=change_array[:N]
    t_array=t_array[:N]
    
    bin_array=np.zeros(nbins)
    content_array=np.zeros(nbins)

    t_array=np.reshape(t_array,(nbins,binsize))
    change_array=np.reshape(change_array,(nbins,binsize))
    
    for i in range(nbins):
        bin_array[i]=i*binsize
        content_array[i]=np.sum(change_array[i])
        
    return bin_array,content_array,t_copy,r_array

def edges(G):
    '''Returns the Number and Types of All Edges
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Return
    ______
    
    cc_edges,dd_edges,cd_edges : float
        Number of edges between cooperators and cooperators, defectors and
        defectors, cooperators and deffectors (or vice versa)
    '''
    cc_edges=0
    dd_edges=0
    cd_edges=0
    
    #nC=cooperators(G)
    #nD=G.number_of_nodes-nC
    cooperators=[]
    defectors=[]
    
    for i in G.nodes:
        if G.nodes[i]["Player"].strategy==1:
            cooperators.append(i)
        else:
            defectors.append(i)
        
    edges=np.array(G.edges)
    
    for i in range(int(edges.size/2)):
        element=edges[i].tolist()
        if all(x in cooperators for x in element):
            cc_edges+=1
        elif all(x in defectors for x in element):
            dd_edges+=1
        else: cd_edges+=1
    
    return cc_edges,dd_edges,cd_edges

def local_freq(G,n):
    '''Returns the Local Frequency of Cooperators of Node n
    
    Calculate the proportion of cooperators among all neighbours of node n
    
    Parameters
    __________
    
    G : networkx.network
        Network
    n : int
        Index of node to evaluate
        
    Returns
    _______
    
    strats/n_neighbours : float
        Loacal frequency
    '''
    neighbours=G.nodes[n]["Player"].neighbour_array
    #print(neighbours)
    n_neighbours=neighbours.size
    strats=0
    for i in neighbours:
        strats+=G.nodes[i]["Player"].strategy
        #print(G.nodes[i]["Player"].strategy)
    return strats/n_neighbours

def local_frequencies(G):
    '''Array of Local Frequencies of all Nodes
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    c_freqs,d_freqs : np.array
        Arrays of the local frequencies of all cooperators and defectors, 
        using "local freq"
    '''
    cooperators=[]
    defectors=[]
    
    c_freqs=[]
    d_freqs=[]

    for i in G.nodes:
        node=G.nodes[i]["Player"]
        #strategies=0
        
        if node.strategy==1:
            cooperators.append(i)
            c_freqs.append(local_freq(G,i))
    
        else:
            defectors.append(i)
            d_freqs.append(local_freq(G,i))

            
    return c_freqs,d_freqs

def captured_nodes(G):
    '''Number of Defectors Surrounded by Defectors
        and Cooperators by Cooperators
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    capt_defs : int
        Number of defectors in G who have only defectors as neighbours
    Dindexes : np.array
        Indices of such defectors
    capt_coops : int
        Number of cooperators in G who have only cooperators as neighbours
    Cindexes : np.array
        Indices of such cooperators
    '''
    capt_defs=0
    capt_coops=0
    Dindexes=[]
    Cindexes=[]
    for i in G.nodes:
        strategies=0
        node=G.nodes[i]["Player"]
        neighbour_array=node.neighbour_array
        #neighbour_array=neighbours(G,i)
        if node.strategy==0:
            for j in range(neighbour_array.size):
                if G.nodes[neighbour_array[j]]["Player"].strategy==1: break
                else: strategies+=1
            if strategies==neighbour_array.size: 
                capt_defs+=1
                Dindexes.append(i)
        if node.strategy==1:
            for j in range(neighbour_array.size):
                if G.nodes[neighbour_array[j]]["Player"].strategy==0: break
                else: strategies+=1
            if strategies==neighbour_array.size: 
                capt_coops+=1
                Cindexes.append(i)
    return capt_defs,Dindexes,capt_coops,Cindexes

def defectors(G):
    '''Return a List of the Indices of all Defectors
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    defector_list : list
        List of indices of defectors in network
    
    '''
    defector_list=[]
    for i in G.nodes:
        if G.nodes[i]["Player"].strategy==0:
            defector_list.append(i)
    return defector_list

def susceptible_nodes(G,ntype):
    '''Return the Numbers of Cooperators and Defectors who could change in 
    RHPM any given updatingprocedure
    
    Parameters
    __________
    
    G : networkx.network
        Network
    ntype : int
        Type of updating procedure
        
        
    Returns
    _______
    
    sus_coops : int
        Number of Cooperators who have at least one Defector in their 
        neighbourhood
    sus_defs : int
        Number of Defectors who have at least one Cooperator in their 
        neighbourhood
    
    '''
    sus_defs=0
    sus_coops=0
    if ntype==1 or ntype==2:
        for i in G.nodes():
            neighbour_array=G.nodes[i]["Player"].neighbour_array
            if G.nodes[i]["Player"].strategy==0:
                for j in range(neighbour_array.size):
                    if G.nodes[neighbour_array[j]]["Player"].strategy==1: 
                        sus_defs+=1
                        break
            else:
                for j in range(neighbour_array.size):
                    if G.nodes[neighbour_array[j]]["Player"].strategy==0: 
                        sus_coops+=1
                        break
        sus_coops/=1
        sus_defs/=1

    if ntype==2:
        D_s,blabla,C_s,blabla=captured_nodes(G)
        sus_defs+=D_s
        sus_coops+=C_s
        
    if ntype==3:
        sus_coops=cooperators(G)
        sus_defs=G.number_of_nodes()-sus_coops
        
    if ntype==4:
        D_s,blabla,C_s,blabla=captured_nodes(G)
        sus_coops+=C_s
        for i in G.nodes():
            if G.degree[i]==0:
                neighbour_array=G.nodes[i]["Player"].neighbour_array
                if sum(neighbour_array)!=neighbour_array.size:
                    sus_defs+=1
                else: break
        
        
    return sus_coops,sus_defs

def avg_susceptible_nodes(G,e_C,e_D,w,ntype):
     '''Return the average (after 25 iterations) 
     Numbers of Cooperators and Defectors who could 
     change in the RHPM updating procedure.
    
    Parameters
    __________
    
    G : networkx.network
        Network
    e_C : float
        Normalised Cooperator Harvesting Effort
    e_D : float
        Normalised Defector Harvesting Effort
    w : float
        Marginal cost of effort
        
    Returns
    _______
    
    C_array : np.array
        Arrayof the numbers of susceptibel cooperators
    D_array : np.array
        Arrayof the numbers of susceptibel Defectors
    '''
    
    C_array=np.array([])
    D_array=np.array([])
    
    R=w
    for i in range(25):
        G=initialise(G,int(0.5*G.number_of_nodes()))

        t,r,x=propagate(300,G,w,e_C,e_D,w,1,ntype)
            
        C,D=susceptible_nodes(G,ntype)
        C_array=np.append(C_array,C)
        D_array=np.append(D_array,D)
    
    return C_array,D_array


def prediction1(e_C,e_D,w,N):
     '''Returns the expected position of the the steady state for a system with
     the give parameters using the FA model
    
    Parameters
    __________
    
    e_C : float
        Normalised Cooperator Harvesting Effort
    e_D : float
        Normalised Defector Harvesting Effort
    w : float
        Marginal cost of effort
    N : int
        Number of players
        
    Returns
    _______
    
    w+dw : float
        Expected steady state resource level
    
    '''
    
    
    dw=(1/N)*(e_D-e_C)*(1-(2*(1-w-e_D))/(e_C-e_D))
    return w+dw

def prediction2(e_C,e_D,w,N,Cs,Ds,eCs,eDs):
    '''Returns the expected position of the the steady state for a system with
     the give parameters using the SA model
    
    Parameters
    __________
    
    e_C : float
        Normalised Cooperator Harvesting Effort
    e_D : float
        Normalised Defector Harvesting Effort
    w : float
        Marginal cost of effort
    N : int
        Number of players
    Cs,Ds : float
        Average numbers of susceptible cooperators and defectors
    eCs,eDs : float
        Standard deviations associated with Cs and Ds
        
    Returns
    _______
    
    w+dw : float
        Expected steady state resource level
    edw : float
        Associated error
    
    '''
    edw=((1/N)*(e_D-e_C))*((np.sqrt(eCs**2+eDs**2))/(N))
    if Ds==0 and Cs==0:
        dw=0
    elif Ds==0:
        dw=((1/N)*(e_D-e_C))
    elif Cs==0:
        dw=((1/N)*(e_D-e_C))
    else:
        dw=((1/N)*(e_D-e_C))*((Ds-Cs)/(N))
        
    return w+dw,edw

def mean_field_outcome(N,w,e_C,e_D,R_i,C_i):
    '''Returns the resource level and cooperator frction of a system simulated 
    for 300 time steps
    
    Parameters
    __________
    
    e_C : float
        Normalised Cooperator Harvesting Effort
    e_D : float
        Normalised Defector Harvesting Effort
    w : float
        Marginal cost of effort
    N : int
        Number of players
    R_i : float
        Initial resource level
    C_i : int
        Initial number of cooperators
        
    Returns
    _______
    
    r : int
        Resource level at t=300
    x : int
        Resource level at t=300
    
    '''
    t,r,x=mf.continuous_mean_field(300,N,w,e_C,e_D,R_i,C_i)
    return r[299],x[299]

def average_species_degree(G):
    '''Return the average degree of cooperators and defectors of a network and 
    also the number of edges between defectors
    
    Parameters
    __________
    
    G : networkx.network
        Network
        
    Returns
    _______
    
    avg_C_degree : float
        Average cooperator degree
    avg_D_degree : float
        Average defector degree
    DD : int
        Number of edges between defectors
    
    '''
    D_degree=0
    C_degree=0
    C_freq=0
    for i in G.nodes:
        if G.nodes[i]["Player"].strategy==0:
            D_degree+=G.degree[i]
        elif G.nodes[i]["Player"].strategy==1:
            C_degree+=G.degree[i]
            C_freq+=1
        else: return "Error"
        
    CC,DD,Cd=edges(G)
    
    avg_C_degree=C_degree/C_freq
    avg_D_degree=D_degree/(G.number_of_nodes()-C_freq)
    
    avg_CC_degree=2*CC/C_freq
    avg_DD_degree=2*DD/(G.number_of_nodes()-C_freq)
    
    return avg_C_degree,avg_D_degree,DD


        

    
    
    
    
    
    

   
    