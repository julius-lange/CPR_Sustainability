#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:57:51 2021

@author: juliuslange
"""

import player as pl
import networks as net
import numpy as np

#%%
N=77
e_C=0.7
e_D=1.1
R_i=0.5
C_i=35

N_reps=100
N_grid=100

W_array=np.linspace(0.01,0.35,N_grid)

C_sus1=np.zeros(N_grid)
D_sus1=np.zeros(N_grid)
eC_sus1=np.zeros(N_grid)
eD_sus1=np.zeros(N_grid)

C_sus2=np.zeros(N_grid)
D_sus2=np.zeros(N_grid)
eC_sus2=np.zeros(N_grid)
eD_sus2=np.zeros(N_grid)

for i in range(N_grid):
    G1=net.make_network(0,35,77,0.0868)
    G2=net.make_network(3,35,77)

    C1,D1=pl.avg_susceptible_nodes(G1,e_C,e_D,W_array[i],1)    
    C2,D2=pl.avg_susceptible_nodes(G2,e_C,e_D,W_array[i],1)
    
    C_sus1[i]=np.mean(C1)
    D_sus1[i]=np.mean(D1)
    eC_sus1[i]=np.std(C1)
    eD_sus1[i]=np.std(D1)
    
    C_sus2[i]=np.mean(C2)
    D_sus2[i]=np.mean(D2)
    eC_sus2[i]=np.std(C2)
    eD_sus2[i]=np.std(D2)    
    
    print(i)
    
#%%
    
pred_mf=np.zeros(N_grid) 
   
pred1=array=np.zeros(N_grid)

pred2_random=array=np.zeros(N_grid)
pred2_lesmis=array=np.zeros(N_grid)

epred2_random=array=np.zeros(N_grid)
epred2_lesmis=array=np.zeros(N_grid)

for i in range(N_grid):
    #pred_mf[i]=pl.mean_field_outcome(N,W_array[i],e_C,e_D,R_i,C_i)[0]
    pred_mf[i]=(1-1.1-W_array[i])/(-0.4)
    pred1[i]=1-1.1-pl.prediction1(e_C,e_D,W_array[i],N)*(-0.4)
    pred2_random[i],epred2_random[i]=pl.prediction2(e_C,e_D,W_array[i],N,C_sus1[i],D_sus1[i],eC_sus1[i],eD_sus1[i])
    pred2_lesmis[i],epred2_lesmis[i]=pl.prediction2(e_C,e_D,W_array[i],N,C_sus2[i],D_sus2[i],eC_sus2[i],eD_sus2[i])
    
    
    
    