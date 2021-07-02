#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:13:33 2021

@author: juliuslange
"""

import player as pl
import networks as net
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D



#%%

T=3000
e_C=0.7
e_D=1.1
w=[0.20,0.05]

R_i=0.5
C_i=38

ntype=[1,2,3,4] #Global Prediction
n=1

N=77
n_reps=40

G_random=net.make_network(0,C_i,N,0.0868)
G_lesmis=net.make_network(3,C_i,N,)
G_scalef=net.make_network(1,C_i,N,3)
G_smallw=net.make_network(9,C_i,N,7,0.25)

degree_strats_random1=np.array([])
degree_strats_lesmis1=np.array([])
degree_strats_scalef1=np.array([])
degree_strats_smallw1=np.array([])

dse_random1=np.array([])
dse_lesmis1=np.array([])
dse_scalef1=np.array([])
dse_smallw1=np.array([])

degree_weights_random1=np.array([])
degree_weights_lesmis1=np.array([])
degree_weights_scalef1=np.array([])
degree_weights_smallw1=np.array([])

degrees_random1=np.array([])
degrees_lesmis1=np.array([])
degrees_scalef1=np.array([])
degrees_smallw1=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[0],n,ntype[0])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[0],n,ntype[0])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[0],n,ntype[0])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[0],n,ntype[0])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random1=np.append(degree_strats_random1,b1)
    degree_strats_lesmis1=np.append(degree_strats_lesmis1,b2)
    degree_strats_scalef1=np.append(degree_strats_scalef1,b3)
    degree_strats_smallw1=np.append(degree_strats_smallw1,b4)
    
    dse_random1=np.append(dse_random1,c1)
    dse_lesmis1=np.append(dse_lesmis1,c2)
    dse_scalef1=np.append(dse_scalef1,c3)
    dse_smallw1=np.append(dse_smallw1,c4)
    
    degrees_random1=np.append(degrees_random1,a1)
    degrees_lesmis1=np.append(degrees_lesmis1,a2)
    degrees_scalef1=np.append(degrees_scalef1,a3)
    degrees_smallw1=np.append(degrees_smallw1,a4)
    
    degree_weights_random1=np.append(degree_weights_random1,d1)
    degree_weights_lesmis1=np.append(degree_weights_lesmis1,d2)
    degree_weights_scalef1=np.append(degree_weights_scalef1,d3)
    degree_weights_smallw1=np.append(degree_weights_smallw1,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random1=np.reshape(degree_strats_random1,(n_reps,ndeg_random))
degree_strats_lesmis1=np.reshape(degree_strats_lesmis1,(n_reps,ndeg_lesmis))
degree_strats_scalef1=np.reshape(degree_strats_scalef1,(n_reps,ndeg_scalef))
degree_strats_smallw1=np.reshape(degree_strats_smallw1,(n_reps,ndeg_smallw))

dse_random1=np.std(degree_strats_random1,axis=0)
dse_lesmis1=np.std(degree_strats_lesmis1,axis=0)
dse_scalef1=np.std(degree_strats_scalef1,axis=0)
dse_smallw1=np.std(degree_strats_smallw1,axis=0)

degree_strats_random1=degree_strats_random1.mean(axis=0)
degree_strats_lesmis1=degree_strats_lesmis1.mean(axis=0)
degree_strats_scalef1=degree_strats_scalef1.mean(axis=0)
degree_strats_smallw1=degree_strats_smallw1.mean(axis=0)

degrees_random1=degrees_random1[:ndeg_random]
degrees_lesmis1=degrees_lesmis1[:ndeg_lesmis]
degrees_scalef1=degrees_scalef1[:ndeg_scalef]
degrees_smallw1=degrees_smallw1[:ndeg_smallw]

##########

degree_strats_random2=np.array([])
degree_strats_lesmis2=np.array([])
degree_strats_scalef2=np.array([])
degree_strats_smallw2=np.array([])

dse_random2=np.array([])
dse_lesmis2=np.array([])
dse_scalef2=np.array([])
dse_smallw2=np.array([])

degree_weights_random2=np.array([])
degree_weights_lesmis2=np.array([])
degree_weights_scalef2=np.array([])
degree_weights_smallw2=np.array([])

degrees_random2=np.array([])
degrees_lesmis2=np.array([])
degrees_scalef2=np.array([])
degrees_smallw2=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[1],n,ntype[0])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[1],n,ntype[0])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[1],n,ntype[0])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[1],n,ntype[0])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random2=np.append(degree_strats_random2,b1)
    degree_strats_lesmis2=np.append(degree_strats_lesmis2,b2)
    degree_strats_scalef2=np.append(degree_strats_scalef2,b3)
    degree_strats_smallw2=np.append(degree_strats_smallw2,b4)
    
    dse_random2=np.append(dse_random2,c1)
    dse_lesmis2=np.append(dse_lesmis2,c2)
    dse_scalef2=np.append(dse_scalef2,c3)
    dse_smallw2=np.append(dse_smallw2,c4)
    
    degrees_random2=np.append(degrees_random2,a1)
    degrees_lesmis2=np.append(degrees_lesmis2,a2)
    degrees_scalef2=np.append(degrees_scalef2,a3)
    degrees_smallw2=np.append(degrees_smallw2,a4)
    
    degree_weights_random2=np.append(degree_weights_random2,d1)
    degree_weights_lesmis2=np.append(degree_weights_lesmis2,d2)
    degree_weights_scalef2=np.append(degree_weights_scalef2,d3)
    degree_weights_smallw2=np.append(degree_weights_smallw2,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random2=np.reshape(degree_strats_random2,(n_reps,ndeg_random))
degree_strats_lesmis2=np.reshape(degree_strats_lesmis2,(n_reps,ndeg_lesmis))
degree_strats_scalef2=np.reshape(degree_strats_scalef2,(n_reps,ndeg_scalef))
degree_strats_smallw2=np.reshape(degree_strats_smallw2,(n_reps,ndeg_smallw))

dse_random2=np.std(degree_strats_random2,axis=0)
dse_lesmis2=np.std(degree_strats_lesmis2,axis=0)
dse_scalef2=np.std(degree_strats_scalef2,axis=0)
dse_smallw2=np.std(degree_strats_smallw2,axis=0)

degree_strats_random2=degree_strats_random2.mean(axis=0)
degree_strats_lesmis2=degree_strats_lesmis2.mean(axis=0)
degree_strats_scalef2=degree_strats_scalef2.mean(axis=0)
degree_strats_smallw2=degree_strats_smallw2.mean(axis=0)

degrees_random2=degrees_random2[:ndeg_random]
degrees_lesmis2=degrees_lesmis2[:ndeg_lesmis]
degrees_scalef2=degrees_scalef2[:ndeg_scalef]
degrees_smallw2=degrees_smallw2[:ndeg_smallw]

################

G_random=net.make_network(0,C_i,N,0.0868)
G_lesmis=net.make_network(3,C_i,N,)
G_scalef=net.make_network(1,C_i,N,3)
G_smallw=net.make_network(9,C_i,N,7,0.25)

degree_strats_random3=np.array([])
degree_strats_lesmis3=np.array([])
degree_strats_scalef3=np.array([])
degree_strats_smallw3=np.array([])

dse_random3=np.array([])
dse_lesmis3=np.array([])
dse_scalef3=np.array([])
dse_smallw3=np.array([])

degree_weights_random3=np.array([])
degree_weights_lesmis3=np.array([])
degree_weights_scalef3=np.array([])
degree_weights_smallw3=np.array([])

degrees_random3=np.array([])
degrees_lesmis3=np.array([])
degrees_scalef3=np.array([])
degrees_smallw3=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[0],n,ntype[1])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[0],n,ntype[1])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[0],n,ntype[1])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[0],n,ntype[1])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random3=np.append(degree_strats_random3,b1)
    degree_strats_lesmis3=np.append(degree_strats_lesmis3,b2)
    degree_strats_scalef3=np.append(degree_strats_scalef3,b3)
    degree_strats_smallw3=np.append(degree_strats_smallw3,b4)
    
    dse_random3=np.append(dse_random3,c1)
    dse_lesmis3=np.append(dse_lesmis3,c2)
    dse_scalef3=np.append(dse_scalef3,c3)
    dse_smallw3=np.append(dse_smallw3,c4)
    
    degrees_random3=np.append(degrees_random3,a1)
    degrees_lesmis3=np.append(degrees_lesmis3,a2)
    degrees_scalef3=np.append(degrees_scalef3,a3)
    degrees_smallw3=np.append(degrees_smallw3,a4)
    
    degree_weights_random3=np.append(degree_weights_random3,d1)
    degree_weights_lesmis3=np.append(degree_weights_lesmis3,d2)
    degree_weights_scalef3=np.append(degree_weights_scalef3,d3)
    degree_weights_smallw3=np.append(degree_weights_smallw3,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random3=np.reshape(degree_strats_random3,(n_reps,ndeg_random))
degree_strats_lesmis3=np.reshape(degree_strats_lesmis3,(n_reps,ndeg_lesmis))
degree_strats_scalef3=np.reshape(degree_strats_scalef3,(n_reps,ndeg_scalef))
degree_strats_smallw3=np.reshape(degree_strats_smallw3,(n_reps,ndeg_smallw))

dse_random3=np.std(degree_strats_random3,axis=0)
dse_lesmis3=np.std(degree_strats_lesmis3,axis=0)
dse_scalef3=np.std(degree_strats_scalef3,axis=0)
dse_smallw3=np.std(degree_strats_smallw3,axis=0)

degree_strats_random3=degree_strats_random3.mean(axis=0)
degree_strats_lesmis3=degree_strats_lesmis3.mean(axis=0)
degree_strats_scalef3=degree_strats_scalef3.mean(axis=0)
degree_strats_smallw3=degree_strats_smallw3.mean(axis=0)

degrees_random3=degrees_random3[:ndeg_random]
degrees_lesmis3=degrees_lesmis3[:ndeg_lesmis]
degrees_scalef3=degrees_scalef3[:ndeg_scalef]
degrees_smallw3=degrees_smallw3[:ndeg_smallw]

##########

degree_strats_random4=np.array([])
degree_strats_lesmis4=np.array([])
degree_strats_scalef4=np.array([])
degree_strats_smallw4=np.array([])

dse_random4=np.array([])
dse_lesmis4=np.array([])
dse_scalef4=np.array([])
dse_smallw4=np.array([])

degree_weights_random4=np.array([])
degree_weights_lesmis4=np.array([])
degree_weights_scalef4=np.array([])
degree_weights_smallw4=np.array([])

degrees_random4=np.array([])
degrees_lesmis4=np.array([])
degrees_scalef4=np.array([])
degrees_smallw4=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[1],n,ntype[1])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[1],n,ntype[1])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[1],n,ntype[1])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[1],n,ntype[1])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random4=np.append(degree_strats_random4,b1)
    degree_strats_lesmis4=np.append(degree_strats_lesmis4,b2)
    degree_strats_scalef4=np.append(degree_strats_scalef4,b3)
    degree_strats_smallw4=np.append(degree_strats_smallw4,b4)
    
    dse_random4=np.append(dse_random4,c1)
    dse_lesmis4=np.append(dse_lesmis4,c2)
    dse_scalef4=np.append(dse_scalef4,c3)
    dse_smallw4=np.append(dse_smallw4,c4)
    
    degrees_random4=np.append(degrees_random4,a1)
    degrees_lesmis4=np.append(degrees_lesmis4,a2)
    degrees_scalef4=np.append(degrees_scalef4,a3)
    degrees_smallw4=np.append(degrees_smallw4,a4)
    
    degree_weights_random4=np.append(degree_weights_random4,d1)
    degree_weights_lesmis4=np.append(degree_weights_lesmis4,d2)
    degree_weights_scalef4=np.append(degree_weights_scalef4,d3)
    degree_weights_smallw4=np.append(degree_weights_smallw4,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random4=np.reshape(degree_strats_random4,(n_reps,ndeg_random))
degree_strats_lesmis4=np.reshape(degree_strats_lesmis4,(n_reps,ndeg_lesmis))
degree_strats_scalef4=np.reshape(degree_strats_scalef4,(n_reps,ndeg_scalef))
degree_strats_smallw4=np.reshape(degree_strats_smallw4,(n_reps,ndeg_smallw))

dse_random4=np.std(degree_strats_random4,axis=0)
dse_lesmis4=np.std(degree_strats_lesmis4,axis=0)
dse_scalef4=np.std(degree_strats_scalef4,axis=0)
dse_smallw4=np.std(degree_strats_smallw4,axis=0)

degree_strats_random4=degree_strats_random4.mean(axis=0)
degree_strats_lesmis4=degree_strats_lesmis4.mean(axis=0)
degree_strats_scalef4=degree_strats_scalef4.mean(axis=0)
degree_strats_smallw4=degree_strats_smallw4.mean(axis=0)

degrees_random4=degrees_random4[:ndeg_random]
degrees_lesmis4=degrees_lesmis4[:ndeg_lesmis]
degrees_scalef4=degrees_scalef4[:ndeg_scalef]
degrees_smallw4=degrees_smallw4[:ndeg_smallw]

################

G_random=net.make_network(0,C_i,N,0.0868)
G_lesmis=net.make_network(3,C_i,N,)
G_scalef=net.make_network(1,C_i,N,3)
G_smallw=net.make_network(9,C_i,N,7,0.25)

degree_strats_random5=np.array([])
degree_strats_lesmis5=np.array([])
degree_strats_scalef5=np.array([])
degree_strats_smallw5=np.array([])

dse_random5=np.array([])
dse_lesmis5=np.array([])
dse_scalef5=np.array([])
dse_smallw5=np.array([])

degree_weights_random5=np.array([])
degree_weights_lesmis5=np.array([])
degree_weights_scalef5=np.array([])
degree_weights_smallw5=np.array([])

degrees_random5=np.array([])
degrees_lesmis5=np.array([])
degrees_scalef5=np.array([])
degrees_smallw5=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[0],n,ntype[2])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[0],n,ntype[2])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[0],n,ntype[2])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[0],n,ntype[2])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random5=np.append(degree_strats_random5,b1)
    degree_strats_lesmis5=np.append(degree_strats_lesmis5,b2)
    degree_strats_scalef5=np.append(degree_strats_scalef5,b3)
    degree_strats_smallw5=np.append(degree_strats_smallw5,b4)
    
    dse_random5=np.append(dse_random5,c1)
    dse_lesmis5=np.append(dse_lesmis5,c2)
    dse_scalef5=np.append(dse_scalef5,c3)
    dse_smallw5=np.append(dse_smallw5,c4)
    
    degrees_random5=np.append(degrees_random5,a1)
    degrees_lesmis5=np.append(degrees_lesmis5,a2)
    degrees_scalef5=np.append(degrees_scalef5,a3)
    degrees_smallw5=np.append(degrees_smallw5,a4)
    
    degree_weights_random5=np.append(degree_weights_random5,d1)
    degree_weights_lesmis5=np.append(degree_weights_lesmis5,d2)
    degree_weights_scalef5=np.append(degree_weights_scalef5,d3)
    degree_weights_smallw5=np.append(degree_weights_smallw5,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random5=np.reshape(degree_strats_random5,(n_reps,ndeg_random))
degree_strats_lesmis5=np.reshape(degree_strats_lesmis5,(n_reps,ndeg_lesmis))
degree_strats_scalef5=np.reshape(degree_strats_scalef5,(n_reps,ndeg_scalef))
degree_strats_smallw5=np.reshape(degree_strats_smallw5,(n_reps,ndeg_smallw))

dse_random5=np.std(degree_strats_random5,axis=0)
dse_lesmis5=np.std(degree_strats_lesmis5,axis=0)
dse_scalef5=np.std(degree_strats_scalef5,axis=0)
dse_smallw5=np.std(degree_strats_smallw5,axis=0)

degree_strats_random5=degree_strats_random5.mean(axis=0)
degree_strats_lesmis5=degree_strats_lesmis5.mean(axis=0)
degree_strats_scalef5=degree_strats_scalef5.mean(axis=0)
degree_strats_smallw5=degree_strats_smallw5.mean(axis=0)

degrees_random5=degrees_random5[:ndeg_random]
degrees_lesmis5=degrees_lesmis5[:ndeg_lesmis]
degrees_scalef5=degrees_scalef5[:ndeg_scalef]
degrees_smallw5=degrees_smallw5[:ndeg_smallw]

##########

degree_strats_random6=np.array([])
degree_strats_lesmis6=np.array([])
degree_strats_scalef6=np.array([])
degree_strats_smallw6=np.array([])

dse_random6=np.array([])
dse_lesmis6=np.array([])
dse_scalef6=np.array([])
dse_smallw6=np.array([])

degree_weights_random6=np.array([])
degree_weights_lesmis6=np.array([])
degree_weights_scalef6=np.array([])
degree_weights_smallw6=np.array([])

degrees_random6=np.array([])
degrees_lesmis6=np.array([])
degrees_scalef6=np.array([])
degrees_smallw6=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[1],n,ntype[2])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[1],n,ntype[2])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[1],n,ntype[2])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[1],n,ntype[2])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random6=np.append(degree_strats_random6,b1)
    degree_strats_lesmis6=np.append(degree_strats_lesmis6,b2)
    degree_strats_scalef6=np.append(degree_strats_scalef6,b3)
    degree_strats_smallw6=np.append(degree_strats_smallw6,b4)
    
    dse_random6=np.append(dse_random6,c1)
    dse_lesmis6=np.append(dse_lesmis6,c2)
    dse_scalef6=np.append(dse_scalef6,c3)
    dse_smallw6=np.append(dse_smallw6,c4)
    
    degrees_random6=np.append(degrees_random6,a1)
    degrees_lesmis6=np.append(degrees_lesmis6,a2)
    degrees_scalef6=np.append(degrees_scalef6,a3)
    degrees_smallw6=np.append(degrees_smallw6,a4)
    
    degree_weights_random6=np.append(degree_weights_random6,d1)
    degree_weights_lesmis6=np.append(degree_weights_lesmis6,d2)
    degree_weights_scalef6=np.append(degree_weights_scalef6,d3)
    degree_weights_smallw6=np.append(degree_weights_smallw6,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random6=np.reshape(degree_strats_random6,(n_reps,ndeg_random))
degree_strats_lesmis6=np.reshape(degree_strats_lesmis6,(n_reps,ndeg_lesmis))
degree_strats_scalef6=np.reshape(degree_strats_scalef6,(n_reps,ndeg_scalef))
degree_strats_smallw6=np.reshape(degree_strats_smallw6,(n_reps,ndeg_smallw))

dse_random6=np.std(degree_strats_random6,axis=0)
dse_lesmis6=np.std(degree_strats_lesmis6,axis=0)
dse_scalef6=np.std(degree_strats_scalef6,axis=0)
dse_smallw6=np.std(degree_strats_smallw6,axis=0)

degree_strats_random6=degree_strats_random6.mean(axis=0)
degree_strats_lesmis6=degree_strats_lesmis6.mean(axis=0)
degree_strats_scalef6=degree_strats_scalef6.mean(axis=0)
degree_strats_smallw6=degree_strats_smallw6.mean(axis=0)

degrees_random6=degrees_random6[:ndeg_random]
degrees_lesmis6=degrees_lesmis6[:ndeg_lesmis]
degrees_scalef6=degrees_scalef6[:ndeg_scalef]
degrees_smallw6=degrees_smallw6[:ndeg_smallw]

################

G_random=net.make_network(0,C_i,N,0.0868)
G_lesmis=net.make_network(3,C_i,N,)
G_scalef=net.make_network(1,C_i,N,3)
G_smallw=net.make_network(9,C_i,N,7,0.25)

degree_strats_random7=np.array([])
degree_strats_lesmis7=np.array([])
degree_strats_scalef7=np.array([])
degree_strats_smallw7=np.array([])

dse_random7=np.array([])
dse_lesmis7=np.array([])
dse_scalef7=np.array([])
dse_smallw7=np.array([])

degree_weights_random7=np.array([])
degree_weights_lesmis7=np.array([])
degree_weights_scalef7=np.array([])
degree_weights_smallw7=np.array([])

degrees_random7=np.array([])
degrees_lesmis7=np.array([])
degrees_scalef7=np.array([])
degrees_smallw7=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[0],n,ntype[3])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[0],n,ntype[3])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[0],n,ntype[3])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[0],n,ntype[3])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random7=np.append(degree_strats_random7,b1)
    degree_strats_lesmis7=np.append(degree_strats_lesmis7,b2)
    degree_strats_scalef7=np.append(degree_strats_scalef7,b3)
    degree_strats_smallw7=np.append(degree_strats_smallw7,b4)
    
    dse_random7=np.append(dse_random7,c1)
    dse_lesmis7=np.append(dse_lesmis7,c2)
    dse_scalef7=np.append(dse_scalef7,c3)
    dse_smallw7=np.append(dse_smallw7,c4)
    
    degrees_random7=np.append(degrees_random7,a1)
    degrees_lesmis7=np.append(degrees_lesmis7,a2)
    degrees_scalef7=np.append(degrees_scalef7,a3)
    degrees_smallw7=np.append(degrees_smallw7,a4)
    
    degree_weights_random7=np.append(degree_weights_random7,d1)
    degree_weights_lesmis7=np.append(degree_weights_lesmis7,d2)
    degree_weights_scalef7=np.append(degree_weights_scalef7,d3)
    degree_weights_smallw7=np.append(degree_weights_smallw7,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random7=np.reshape(degree_strats_random7,(n_reps,ndeg_random))
degree_strats_lesmis7=np.reshape(degree_strats_lesmis7,(n_reps,ndeg_lesmis))
degree_strats_scalef7=np.reshape(degree_strats_scalef7,(n_reps,ndeg_scalef))
degree_strats_smallw7=np.reshape(degree_strats_smallw7,(n_reps,ndeg_smallw))

dse_random7=np.std(degree_strats_random7,axis=0)
dse_lesmis7=np.std(degree_strats_lesmis7,axis=0)
dse_scalef7=np.std(degree_strats_scalef7,axis=0)
dse_smallw7=np.std(degree_strats_smallw7,axis=0)

degree_strats_random7=degree_strats_random7.mean(axis=0)
degree_strats_lesmis7=degree_strats_lesmis7.mean(axis=0)
degree_strats_scalef7=degree_strats_scalef7.mean(axis=0)
degree_strats_smallw7=degree_strats_smallw7.mean(axis=0)

degrees_random7=degrees_random7[:ndeg_random]
degrees_lesmis7=degrees_lesmis7[:ndeg_lesmis]
degrees_scalef7=degrees_scalef7[:ndeg_scalef]
degrees_smallw7=degrees_smallw7[:ndeg_smallw]

##########

degree_strats_random8=np.array([])
degree_strats_lesmis8=np.array([])
degree_strats_scalef8=np.array([])
degree_strats_smallw8=np.array([])

dse_random8=np.array([])
dse_lesmis8=np.array([])
dse_scalef8=np.array([])
dse_smallw8=np.array([])

degree_weights_random8=np.array([])
degree_weights_lesmis8=np.array([])
degree_weights_scalef8=np.array([])
degree_weights_smallw8=np.array([])

degrees_random8=np.array([])
degrees_lesmis8=np.array([])
degrees_scalef8=np.array([])
degrees_smallw8=np.array([])

for i in range(n_reps):
    data1=pl.propagate(T,G_random,R_i,e_C,e_D,w[1],n,ntype[3])
    data2=pl.propagate(T,G_lesmis,R_i,e_C,e_D,w[1],n,ntype[3])
    data3=pl.propagate(T,G_scalef,R_i,e_C,e_D,w[1],n,ntype[3])
    data4=pl.propagate(T,G_smallw,R_i,e_C,e_D,w[1],n,ntype[3])
    
    a1,b1,c1,d1=pl.average_degree_strategy_array(G_random,T)
    a2,b2,c2,d2=pl.average_degree_strategy_array(G_lesmis,T)
    a3,b3,c3,d3=pl.average_degree_strategy_array(G_scalef,T)
    a4,b4,c4,d4=pl.average_degree_strategy_array(G_smallw,T)
    
    ndeg_random=a1.size
    ndeg_lesmis=a2.size
    ndeg_scalef=a3.size
    ndeg_smallw=a4.size
    
    degree_strats_random8=np.append(degree_strats_random8,b1)
    degree_strats_lesmis8=np.append(degree_strats_lesmis8,b2)
    degree_strats_scalef8=np.append(degree_strats_scalef8,b3)
    degree_strats_smallw8=np.append(degree_strats_smallw8,b4)
    
    dse_random8=np.append(dse_random8,c1)
    dse_lesmis8=np.append(dse_lesmis8,c2)
    dse_scalef8=np.append(dse_scalef8,c3)
    dse_smallw8=np.append(dse_smallw8,c4)
    
    degrees_random8=np.append(degrees_random8,a1)
    degrees_lesmis8=np.append(degrees_lesmis8,a2)
    degrees_scalef8=np.append(degrees_scalef8,a3)
    degrees_smallw8=np.append(degrees_smallw8,a4)
    
    degree_weights_random8=np.append(degree_weights_random8,d1)
    degree_weights_lesmis8=np.append(degree_weights_lesmis8,d2)
    degree_weights_scalef8=np.append(degree_weights_scalef8,d3)
    degree_weights_smallw8=np.append(degree_weights_smallw8,d4)
    
    for j in range(77):
        G_random.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_lesmis.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_scalef.nodes[j]["Player"].total_strat=0
    for j in range(77):
        G_smallw.nodes[j]["Player"].total_strat=0
        
    G_random=pl.initialise(G_random,C_i)
    G_lesmis=pl.initialise(G_lesmis,C_i)
    G_scalef=pl.initialise(G_scalef,C_i)
    G_smallw=pl.initialise(G_smallw,C_i)
    
    print(i)
    

degree_strats_random8=np.reshape(degree_strats_random8,(n_reps,ndeg_random))
degree_strats_lesmis8=np.reshape(degree_strats_lesmis8,(n_reps,ndeg_lesmis))
degree_strats_scalef8=np.reshape(degree_strats_scalef8,(n_reps,ndeg_scalef))
degree_strats_smallw8=np.reshape(degree_strats_smallw8,(n_reps,ndeg_smallw))

dse_random8=np.std(degree_strats_random8,axis=0)
dse_lesmis8=np.std(degree_strats_lesmis8,axis=0)
dse_scalef8=np.std(degree_strats_scalef8,axis=0)
dse_smallw8=np.std(degree_strats_smallw8,axis=0)

degree_strats_random8=degree_strats_random8.mean(axis=0)
degree_strats_lesmis8=degree_strats_lesmis8.mean(axis=0)
degree_strats_scalef8=degree_strats_scalef8.mean(axis=0)
degree_strats_smallw8=degree_strats_smallw8.mean(axis=0)

degrees_random8=degrees_random8[:ndeg_random]
degrees_lesmis8=degrees_lesmis8[:ndeg_lesmis]
degrees_scalef8=degrees_scalef8[:ndeg_scalef]
degrees_smallw8=degrees_smallw8[:ndeg_smallw]

################

#%%

def my_function(degree,strats,error):
    z = np.polyfit(np.log(degree), strats, 1, w=error)
    p = np.poly1d(z)
    #print(p)
    y=p(np.log(degree))
    return y
    
#%%

fig,ax=plt.subplots()

ax.plot(degrees_random1,degree_strats_random1,"r+",label="Random Network", alpha=0.7)
ax.plot(degrees_lesmis1,degree_strats_lesmis1,"b+",label="Les Mis Network", alpha=0.7)
ax.plot(degrees_scalef1,degree_strats_scalef1,"g+",label="SF Network", alpha=0.7)
ax.plot(degrees_smallw1,degree_strats_smallw1,"y+",label="SM Network", alpha=0.7)
ax.plot(degrees_lesmis1,1+degrees_lesmis1*0,"k", alpha=0.7)

ax.errorbar(degrees_random1,degree_strats_random1,yerr=dse_random1,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_lesmis1,degree_strats_lesmis1,yerr=dse_lesmis1,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_scalef1,degree_strats_scalef1,yerr=dse_scalef1,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_smallw1,degree_strats_smallw1,yerr=dse_smallw1,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)

ax.plot(degrees_random2,degree_strats_random2,"rx",label="Random Network", alpha=0.7)
ax.plot(degrees_lesmis2,degree_strats_lesmis2,"bx",label="Les Mis Network", alpha=0.7)
ax.plot(degrees_scalef2,degree_strats_scalef2,"gx",label="SF Network", alpha=0.7)
ax.plot(degrees_smallw2,degree_strats_smallw2,"yx",label="SM Network", alpha=0.7)

ax.errorbar(degrees_random2,degree_strats_random2,yerr=dse_random2,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_lesmis2,degree_strats_lesmis2,yerr=dse_lesmis2,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_scalef2,degree_strats_scalef2,yerr=dse_scalef2,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax.errorbar(degrees_smallw2,degree_strats_smallw2,yerr=dse_smallw2,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)


ax.set_xlabel("Degree k")
ax.set_ylabel("Average Strategy x")

ax.semilogx()

plt.legend()
plt.tight_layout()
plt.show()

#%%

fig,ax=plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
ax[0,0].semilogx()


ax[0,0].plot(degrees_random1,degree_strats_random1,"ro",label="Random Network", alpha=0.7)
ax[0,0].plot(degrees_lesmis1,degree_strats_lesmis1,"bo",label="Les Mis Network", alpha=0.7)
ax[0,0].plot(degrees_scalef1,degree_strats_scalef1,"go",label="SF Network", alpha=0.7)
ax[0,0].plot(degrees_smallw1,degree_strats_smallw1,"yo",label="SM Network", alpha=0.7)
#ax[0,0].plot(degrees_lesmis1,1+degrees_lesmis1*0,"k", alpha=0.7)

ax[0,0].errorbar(degrees_random1,degree_strats_random1,yerr=dse_random1,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_lesmis1,degree_strats_lesmis1,yerr=dse_lesmis1,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_scalef1,degree_strats_scalef1,yerr=dse_scalef1,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_smallw1,degree_strats_smallw1,yerr=dse_smallw1,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)

ax[0,0].plot(degrees_random2,degree_strats_random2,"rx",label="Random Network", alpha=0.7)
ax[0,0].plot(degrees_lesmis2,degree_strats_lesmis2,"bx",label="Les Mis Network", alpha=0.7)
ax[0,0].plot(degrees_scalef2,degree_strats_scalef2,"gx",label="SF Network", alpha=0.7)
ax[0,0].plot(degrees_smallw2,degree_strats_smallw2,"yx",label="SM Network", alpha=0.7)

ax[0,0].errorbar(degrees_random2,degree_strats_random2,yerr=dse_random2,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_lesmis2,degree_strats_lesmis2,yerr=dse_lesmis2,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_scalef2,degree_strats_scalef2,yerr=dse_scalef2,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].errorbar(degrees_smallw2,degree_strats_smallw2,yerr=dse_smallw2,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,0].set_title("RHPM")

ax[0,1].plot(degrees_random3,degree_strats_random3,"ro",label="Random Network", alpha=0.7)
ax[0,1].plot(degrees_lesmis3,degree_strats_lesmis3,"bo",label="Les Mis Network", alpha=0.7)
ax[0,1].plot(degrees_scalef3,degree_strats_scalef3,"go",label="SF Network", alpha=0.7)
ax[0,1].plot(degrees_smallw3,degree_strats_smallw3,"yo",label="SM Network", alpha=0.7)
#ax[0,1].plot(degrees_lesmis3,1+degrees_lesmis3*0,"k", alpha=0.7)

ax[0,1].errorbar(degrees_random3,degree_strats_random3,yerr=dse_random3,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_lesmis3,degree_strats_lesmis3,yerr=dse_lesmis3,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_scalef3,degree_strats_scalef3,yerr=dse_scalef3,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_smallw3,degree_strats_smallw3,yerr=dse_smallw3,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)

ax[0,1].plot(degrees_random4,degree_strats_random4,"rx",label="Random Network", alpha=0.7)
ax[0,1].plot(degrees_lesmis4,degree_strats_lesmis4,"bx",label="Les Mis Network", alpha=0.7)
ax[0,1].plot(degrees_scalef4,degree_strats_scalef4,"gx",label="SF Network", alpha=0.7)
ax[0,1].plot(degrees_smallw4,degree_strats_smallw4,"yx",label="SM Network", alpha=0.7)

ax[0,1].errorbar(degrees_random4,degree_strats_random4,yerr=dse_random4,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_lesmis4,degree_strats_lesmis4,yerr=dse_lesmis4,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_scalef4,degree_strats_scalef4,yerr=dse_scalef4,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].errorbar(degrees_smallw4,degree_strats_smallw4,yerr=dse_smallw4,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[0,1].set_title("IHPM")

ax[1,0].plot(degrees_random5,degree_strats_random5,"ro",label="Random Network", alpha=0.7)
ax[1,0].plot(degrees_lesmis5,degree_strats_lesmis5,"bo",label="Les Mis Network", alpha=0.7)
ax[1,0].plot(degrees_scalef5,degree_strats_scalef5,"go",label="SF Network", alpha=0.7)
ax[1,0].plot(degrees_smallw5,degree_strats_smallw5,"yo",label="SM Network", alpha=0.7)
#ax[1,0].plot(degrees_lesmis5,1+degrees_lesmis5*0,"k", alpha=0.7)

ax[1,0].errorbar(degrees_random5,degree_strats_random5,yerr=dse_random5,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_lesmis5,degree_strats_lesmis5,yerr=dse_lesmis5,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_scalef5,degree_strats_scalef5,yerr=dse_scalef5,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_smallw5,degree_strats_smallw5,yerr=dse_smallw5,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)

ax[1,0].plot(degrees_random6,degree_strats_random6,"rx",label="Random Network", alpha=0.7)
ax[1,0].plot(degrees_lesmis6,degree_strats_lesmis6,"bx",label="Les Mis Network", alpha=0.7)
ax[1,0].plot(degrees_scalef6,degree_strats_scalef6,"gx",label="SF Network", alpha=0.7)
ax[1,0].plot(degrees_smallw6,degree_strats_smallw6,"yx",label="SM Network", alpha=0.7)

ax[1,0].errorbar(degrees_random6,degree_strats_random6,yerr=dse_random6,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_lesmis6,degree_strats_lesmis6,yerr=dse_lesmis6,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_scalef6,degree_strats_scalef6,yerr=dse_scalef6,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].errorbar(degrees_smallw6,degree_strats_smallw6,yerr=dse_smallw6,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,0].set_title("GP")

ax[1,1].plot(degrees_random7,degree_strats_random7,"ro",label="Random Network", alpha=0.7)
ax[1,1].plot(degrees_lesmis7,degree_strats_lesmis7,"bo",label="Les Mis Network", alpha=0.7)
ax[1,1].plot(degrees_scalef7,degree_strats_scalef7,"go",label="SF Network", alpha=0.7)
ax[1,1].plot(degrees_smallw7,degree_strats_smallw7,"yo",label="SM Network", alpha=0.7)
#ax[1,1].plot(degrees_lesmis7,1+degrees_lesmis7*0,"k", alpha=0.7)

ax[1,1].errorbar(degrees_random7,degree_strats_random7,yerr=dse_random7,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_lesmis7,degree_strats_lesmis7,yerr=dse_lesmis7,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_scalef7,degree_strats_scalef7,yerr=dse_scalef7,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_smallw7,degree_strats_smallw7,yerr=dse_smallw7,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)

ax[1,1].plot(degrees_random8,degree_strats_random8,"rx",label="Random Network", alpha=0.7)
ax[1,1].plot(degrees_lesmis8,degree_strats_lesmis8,"bx",label="Les Mis Network", alpha=0.7)
ax[1,1].plot(degrees_scalef8,degree_strats_scalef8,"gx",label="SF Network", alpha=0.7)
ax[1,1].plot(degrees_smallw8,degree_strats_smallw8,"yx",label="SM Network", alpha=0.7)

ax[1,1].errorbar(degrees_random8,degree_strats_random8,yerr=dse_random8,linestyle='none',ecolor="r",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_lesmis8,degree_strats_lesmis8,yerr=dse_lesmis8,linestyle='none',ecolor="b",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_scalef8,degree_strats_scalef8,yerr=dse_scalef8,linestyle='none',ecolor="g",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].errorbar(degrees_smallw8,degree_strats_smallw8,yerr=dse_smallw8,linestyle='none',ecolor="y",elinewidth=1,barsabove=True,capsize=3,alpha=0.7)
ax[1,1].set_title("LP")

fig.text(0.5, 0.13,'Degree, $k$', ha='center', va='center')
fig.text(0.05, 0.5,r'Average Strategy, $\langle s \rangle$', ha='center', va='center', rotation='vertical')

ax[0,0].set_ylim([0,1.0])

legend_elements = [Line2D([0], [0], color='r', marker='.', label='Random', ls='None'),
                   Line2D([0], [0], color='b', marker='.', label='LM', ls='None'),
                   Line2D([0], [0], color='g', marker='.', label='SF', ls='None'),
                   Line2D([0], [0], color='y', marker='.', label='SM', ls='None'),
                   Line2D([0], [0], color='k', marker='x', label='w=0.05', ls='None'),
                   Line2D([0], [0], color='k', marker='o', label='w=0.20', ls='None')]

plt.legend(handles=legend_elements,loc="lower center",bbox_to_anchor=(-0.1,-0.60),ncol=3)
fig.subplots_adjust(bottom=0.20)

plt.show()
plt.savefig("/Users/juliuslange/Desktop/Report_Graphics/Degree_strat.png")

#%%

print("PS w=0.20 \n")

print("degrees_sf=",degrees_random)
print("degree_strats_sf=",degree_strats_random)
print("degree_strats_sf=",dse_random,"\n")

print("degrees_sf=",degrees_lesmis)
print("degree_strats_sf=",degree_strats_lesmis)
print("degree_strats_sf=",dse_lesmis,"\n")

print("degrees_sf=",degrees_sf)
print("degree_strats_sf=",degree_strats_sf)
print("degree_strats_sf=",dse_sf,"\n")

print("degrees_sf=",degrees_sm)
print("degree_strats_sf=",degree_strats_sm)
print("degree_strats_sf=",dse_sm,"\n")

