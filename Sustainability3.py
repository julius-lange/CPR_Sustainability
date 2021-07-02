#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:29:22 2021

@author: juliuslange
"""
import player as pl
import networks as net
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

#%%

def frequency_of_change_events(x_array):
    change_events=0
    for i in range(x_array.size-1):
        if x_array[i]!=x_array[i+1]:
            change_events+=1
    return change_events/x_array.size

def total_payoff(G):
    total_payoff=0
    
    for i in G.nodes:
        total_payoff+=G.nodes[i]["Player"].total_payoff
    
    return total_payoff

def steady_state_R(r_array,T_cut):
    return np.mean(r_array[T_cut:])
    

#%%

'''
Sustainability Comparison 3

Observe <R> , P_tot, and FoC of Random, LM, SF, and SM networks against
different choices of w
'''
N_grid=100
w_array=np.linspace(0,0.35,N_grid)
T=3000
e_C=0.7
e_D=1.1
n=1
p=1
N=77
R_i=0.5
C_i=0.5
T_cut=500
N_reps=10

ntype=np.array([1,2,3,4]) #Updating Procedure

G1=net.make_network(0,int(N*C_i),N,0.0868)
G2=net.make_network(3,int(N*C_i))
G3=net.make_network(1,int(N*C_i),N,3)
G4=net.make_network(9,int(N*C_i),N,7,0.25)

print("RHPM")

R_random1=np.zeros(N_grid)
R_lesmis1=np.zeros(N_grid)
R_scalef1=np.zeros(N_grid)
R_smallw1=np.zeros(N_grid)

P_random1=np.zeros(N_grid)
P_lesmis1=np.zeros(N_grid)
P_scalef1=np.zeros(N_grid)
P_smallw1=np.zeros(N_grid)

F_random1=np.zeros(N_grid)
F_lesmis1=np.zeros(N_grid)
F_scalef1=np.zeros(N_grid)
F_smallw1=np.zeros(N_grid)

eR_random1=np.zeros(N_grid)
eR_lesmis1=np.zeros(N_grid)
eR_scalef1=np.zeros(N_grid)
eR_smallw1=np.zeros(N_grid)

eP_random1=np.zeros(N_grid)
eP_lesmis1=np.zeros(N_grid)
eP_scalef1=np.zeros(N_grid)
eP_smallw1=np.zeros(N_grid)

eF_random1=np.zeros(N_grid)
eF_lesmis1=np.zeros(N_grid)
eF_scalef1=np.zeros(N_grid)
eF_smallw1=np.zeros(N_grid)

for i in range(w_array.size):
    
    p1=np.zeros(N_reps)
    p2=np.zeros(N_reps)
    p3=np.zeros(N_reps)
    p4=np.zeros(N_reps)

    r1=np.zeros(N_reps)
    r2=np.zeros(N_reps)
    r3=np.zeros(N_reps)
    r4=np.zeros(N_reps)
    
    f1=np.zeros(N_reps)
    f2=np.zeros(N_reps)
    f3=np.zeros(N_reps)
    f4=np.zeros(N_reps)

    for j in range(N_reps):
        
        t_array1,r_array1,x_array1=pl.propagate(T,G1,R_i,e_C,e_D,w_array[i],n,ntype[0])
        t_array2,r_array2,x_array2=pl.propagate(T,G2,R_i,e_C,e_D,w_array[i],n,ntype[0])
        t_array3,r_array3,x_array3=pl.propagate(T,G3,R_i,e_C,e_D,w_array[i],n,ntype[0])
        t_array4,r_array4,x_array4=pl.propagate(T,G4,R_i,e_C,e_D,w_array[i],n,ntype[0])
        
        p1[j]=total_payoff(G1)
        p2[j]=total_payoff(G2)
        p3[j]=total_payoff(G3)
        p4[j]=total_payoff(G4)
        
        r1[j]=steady_state_R(r_array1,T_cut)
        r2[j]=steady_state_R(r_array2,T_cut)
        r3[j]=steady_state_R(r_array3,T_cut)
        r4[j]=steady_state_R(r_array4,T_cut)
        
        f1[j]=frequency_of_change_events(x_array1)
        f2[j]=frequency_of_change_events(x_array2)
        f3[j]=frequency_of_change_events(x_array3)
        f4[j]=frequency_of_change_events(x_array4)

        G1=pl.initialise(G1,int(N*C_i))
        G2=pl.initialise(G2,int(N*C_i))
        G3=pl.initialise(G3,int(N*C_i))
        G4=pl.initialise(G4,int(N*C_i))

        #print("\t",j)
        
    R_random1[i]=np.mean(r1)
    R_lesmis1[i]=np.mean(r2)
    R_scalef1[i]=np.mean(r3)
    R_smallw1[i]=np.mean(r4)
    
    P_random1[i]=np.mean(p1)
    P_lesmis1[i]=np.mean(p2)
    P_scalef1[i]=np.mean(p3)
    P_smallw1[i]=np.mean(p4)
    
    F_random1[i]=np.mean(f1)
    F_lesmis1[i]=np.mean(f2)
    F_scalef1[i]=np.mean(f3)
    F_smallw1[i]=np.mean(f4)
    
    eR_random1[i]=np.std(r1)
    eR_lesmis1[i]=np.std(r2)
    eR_scalef1[i]=np.std(r3)
    eR_smallw1[i]=np.std(r4)
    
    eP_random1[i]=np.std(p1)
    eP_lesmis1[i]=np.std(p2)
    eP_scalef1[i]=np.std(p3)
    eP_smallw1[i]=np.std(p4)
    
    eF_random1[i]=np.std(f1)
    eF_lesmis1[i]=np.std(f2)
    eF_scalef1[i]=np.std(f3)
    eF_smallw1[i]=np.std(f4)
    
    print(i)
    
    
print("IHPM")
    
R_random2=np.zeros(N_grid)
R_lesmis2=np.zeros(N_grid)
R_scalef2=np.zeros(N_grid)
R_smallw2=np.zeros(N_grid)

P_random2=np.zeros(N_grid)
P_lesmis2=np.zeros(N_grid)
P_scalef2=np.zeros(N_grid)
P_smallw2=np.zeros(N_grid)

F_random2=np.zeros(N_grid)
F_lesmis2=np.zeros(N_grid)
F_scalef2=np.zeros(N_grid)
F_smallw2=np.zeros(N_grid)

eR_random2=np.zeros(N_grid)
eR_lesmis2=np.zeros(N_grid)
eR_scalef2=np.zeros(N_grid)
eR_smallw2=np.zeros(N_grid)

eP_random2=np.zeros(N_grid)
eP_lesmis2=np.zeros(N_grid)
eP_scalef2=np.zeros(N_grid)
eP_smallw2=np.zeros(N_grid)

eF_random2=np.zeros(N_grid)
eF_lesmis2=np.zeros(N_grid)
eF_scalef2=np.zeros(N_grid)
eF_smallw2=np.zeros(N_grid)

for i in range(w_array.size):
    
    p1=np.zeros(N_reps)
    p2=np.zeros(N_reps)
    p3=np.zeros(N_reps)
    p4=np.zeros(N_reps)

    r1=np.zeros(N_reps)
    r2=np.zeros(N_reps)
    r3=np.zeros(N_reps)
    r4=np.zeros(N_reps)
    
    f1=np.zeros(N_reps)
    f2=np.zeros(N_reps)
    f3=np.zeros(N_reps)
    f4=np.zeros(N_reps)

    for j in range(N_reps):
        
        t_array1,r_array1,x_array1=pl.propagate(T,G1,R_i,e_C,e_D,w_array[i],n,ntype[1])
        t_array2,r_array2,x_array2=pl.propagate(T,G2,R_i,e_C,e_D,w_array[i],n,ntype[1])
        t_array3,r_array3,x_array3=pl.propagate(T,G3,R_i,e_C,e_D,w_array[i],n,ntype[1])
        t_array4,r_array4,x_array4=pl.propagate(T,G4,R_i,e_C,e_D,w_array[i],n,ntype[1])
        
        p1[j]=total_payoff(G1)
        p2[j]=total_payoff(G2)
        p3[j]=total_payoff(G3)
        p4[j]=total_payoff(G4)
        
        r1[j]=steady_state_R(r_array1,T_cut)
        r2[j]=steady_state_R(r_array2,T_cut)
        r3[j]=steady_state_R(r_array3,T_cut)
        r4[j]=steady_state_R(r_array4,T_cut)
        
        f1[j]=frequency_of_change_events(x_array1)
        f2[j]=frequency_of_change_events(x_array2)
        f3[j]=frequency_of_change_events(x_array3)
        f4[j]=frequency_of_change_events(x_array4)

        G1=pl.initialise(G1,int(N*C_i))
        G2=pl.initialise(G2,int(N*C_i))
        G3=pl.initialise(G3,int(N*C_i))
        G4=pl.initialise(G4,int(N*C_i))

        #print("\t",j)
        
    R_random2[i]=np.mean(r1)
    R_lesmis2[i]=np.mean(r2)
    R_scalef2[i]=np.mean(r3)
    R_smallw2[i]=np.mean(r4)
    
    P_random2[i]=np.mean(p1)
    P_lesmis2[i]=np.mean(p2)
    P_scalef2[i]=np.mean(p3)
    P_smallw2[i]=np.mean(p4)
    
    F_random2[i]=np.mean(f1)
    F_lesmis2[i]=np.mean(f2)
    F_scalef2[i]=np.mean(f3)
    F_smallw2[i]=np.mean(f4)
    
    eR_random2[i]=np.std(r1)
    eR_lesmis2[i]=np.std(r2)
    eR_scalef2[i]=np.std(r3)
    eR_smallw2[i]=np.std(r4)
    
    eP_random2[i]=np.std(p1)
    eP_lesmis2[i]=np.std(p2)
    eP_scalef2[i]=np.std(p3)
    eP_smallw2[i]=np.std(p4)
    
    eF_random2[i]=np.std(f1)
    eF_lesmis2[i]=np.std(f2)
    eF_scalef2[i]=np.std(f3)
    eF_smallw2[i]=np.std(f4)
    
    print(i)
    

    
print("GP")
    
R_random3=np.zeros(N_grid)
R_lesmis3=np.zeros(N_grid)
R_scalef3=np.zeros(N_grid)
R_smallw3=np.zeros(N_grid)

P_random3=np.zeros(N_grid)
P_lesmis3=np.zeros(N_grid)
P_scalef3=np.zeros(N_grid)
P_smallw3=np.zeros(N_grid)

F_random3=np.zeros(N_grid)
F_lesmis3=np.zeros(N_grid)
F_scalef3=np.zeros(N_grid)
F_smallw3=np.zeros(N_grid)

eR_random3=np.zeros(N_grid)
eR_lesmis3=np.zeros(N_grid)
eR_scalef3=np.zeros(N_grid)
eR_smallw3=np.zeros(N_grid)

eP_random3=np.zeros(N_grid)
eP_lesmis3=np.zeros(N_grid)
eP_scalef3=np.zeros(N_grid)
eP_smallw3=np.zeros(N_grid)

eF_random3=np.zeros(N_grid)
eF_lesmis3=np.zeros(N_grid)
eF_scalef3=np.zeros(N_grid)
eF_smallw3=np.zeros(N_grid)

for i in range(w_array.size):
    
    p1=np.zeros(N_reps)
    p2=np.zeros(N_reps)
    p3=np.zeros(N_reps)
    p4=np.zeros(N_reps)

    r1=np.zeros(N_reps)
    r2=np.zeros(N_reps)
    r3=np.zeros(N_reps)
    r4=np.zeros(N_reps)
    
    f1=np.zeros(N_reps)
    f2=np.zeros(N_reps)
    f3=np.zeros(N_reps)
    f4=np.zeros(N_reps)

    for j in range(N_reps):
        
        t_array1,r_array1,x_array1=pl.propagate(T,G1,R_i,e_C,e_D,w_array[i],n,ntype[2])
        t_array2,r_array2,x_array2=pl.propagate(T,G2,R_i,e_C,e_D,w_array[i],n,ntype[2])
        t_array3,r_array3,x_array3=pl.propagate(T,G3,R_i,e_C,e_D,w_array[i],n,ntype[2])
        t_array4,r_array4,x_array4=pl.propagate(T,G4,R_i,e_C,e_D,w_array[i],n,ntype[2])
        
        p1[j]=total_payoff(G1)
        p2[j]=total_payoff(G2)
        p3[j]=total_payoff(G3)
        p4[j]=total_payoff(G4)
        
        r1[j]=steady_state_R(r_array1,T_cut)
        r2[j]=steady_state_R(r_array2,T_cut)
        r3[j]=steady_state_R(r_array3,T_cut)
        r4[j]=steady_state_R(r_array4,T_cut)
        
        f1[j]=frequency_of_change_events(x_array1)
        f2[j]=frequency_of_change_events(x_array2)
        f3[j]=frequency_of_change_events(x_array3)
        f4[j]=frequency_of_change_events(x_array4)

        G1=pl.initialise(G1,int(N*C_i))
        G2=pl.initialise(G2,int(N*C_i))
        G3=pl.initialise(G3,int(N*C_i))
        G4=pl.initialise(G4,int(N*C_i))

        #print("\t",j)
        
    R_random3[i]=np.mean(r1)
    R_lesmis3[i]=np.mean(r2)
    R_scalef3[i]=np.mean(r3)
    R_smallw3[i]=np.mean(r4)
    
    P_random3[i]=np.mean(p1)
    P_lesmis3[i]=np.mean(p2)
    P_scalef3[i]=np.mean(p3)
    P_smallw3[i]=np.mean(p4)
    
    F_random3[i]=np.mean(f1)
    F_lesmis3[i]=np.mean(f2)
    F_scalef3[i]=np.mean(f3)
    F_smallw3[i]=np.mean(f4)
    
    eR_random3[i]=np.std(r1)
    eR_lesmis3[i]=np.std(r2)
    eR_scalef3[i]=np.std(r3)
    eR_smallw3[i]=np.std(r4)
    
    eP_random3[i]=np.std(p1)
    eP_lesmis3[i]=np.std(p2)
    eP_scalef3[i]=np.std(p3)
    eP_smallw3[i]=np.std(p4)
    
    eF_random3[i]=np.std(f1)
    eF_lesmis3[i]=np.std(f2)
    eF_scalef3[i]=np.std(f3)
    eF_smallw3[i]=np.std(f4)
    
    print(i)
    

    
print("LP")
    
R_random4=np.zeros(N_grid)
R_lesmis4=np.zeros(N_grid)
R_scalef4=np.zeros(N_grid)
R_smallw4=np.zeros(N_grid)

P_random4=np.zeros(N_grid)
P_lesmis4=np.zeros(N_grid)
P_scalef4=np.zeros(N_grid)
P_smallw4=np.zeros(N_grid)

F_random4=np.zeros(N_grid)
F_lesmis4=np.zeros(N_grid)
F_scalef4=np.zeros(N_grid)
F_smallw4=np.zeros(N_grid)

eR_random4=np.zeros(N_grid)
eR_lesmis4=np.zeros(N_grid)
eR_scalef4=np.zeros(N_grid)
eR_smallw4=np.zeros(N_grid)

eP_random4=np.zeros(N_grid)
eP_lesmis4=np.zeros(N_grid)
eP_scalef4=np.zeros(N_grid)
eP_smallw4=np.zeros(N_grid)

eF_random4=np.zeros(N_grid)
eF_lesmis4=np.zeros(N_grid)
eF_scalef4=np.zeros(N_grid)
eF_smallw4=np.zeros(N_grid)

for i in range(w_array.size):
    
    p1=np.zeros(N_reps)
    p2=np.zeros(N_reps)
    p3=np.zeros(N_reps)
    p4=np.zeros(N_reps)

    r1=np.zeros(N_reps)
    r2=np.zeros(N_reps)
    r3=np.zeros(N_reps)
    r4=np.zeros(N_reps)
    
    f1=np.zeros(N_reps)
    f2=np.zeros(N_reps)
    f3=np.zeros(N_reps)
    f4=np.zeros(N_reps)

    for j in range(N_reps):
        
        t_array1,r_array1,x_array1=pl.propagate(T,G1,R_i,e_C,e_D,w_array[i],n,ntype[3])
        t_array2,r_array2,x_array2=pl.propagate(T,G2,R_i,e_C,e_D,w_array[i],n,ntype[3])
        t_array3,r_array3,x_array3=pl.propagate(T,G3,R_i,e_C,e_D,w_array[i],n,ntype[3])
        t_array4,r_array4,x_array4=pl.propagate(T,G4,R_i,e_C,e_D,w_array[i],n,ntype[3])
        
        p1[j]=total_payoff(G1)
        p2[j]=total_payoff(G2)
        p3[j]=total_payoff(G3)
        p4[j]=total_payoff(G4)
        
        r1[j]=steady_state_R(r_array1,T_cut)
        r2[j]=steady_state_R(r_array2,T_cut)
        r3[j]=steady_state_R(r_array3,T_cut)
        r4[j]=steady_state_R(r_array4,T_cut)
        
        f1[j]=frequency_of_change_events(x_array1)
        f2[j]=frequency_of_change_events(x_array2)
        f3[j]=frequency_of_change_events(x_array3)
        f4[j]=frequency_of_change_events(x_array4)

        G1=pl.initialise(G1,int(N*C_i))
        G2=pl.initialise(G2,int(N*C_i))
        G3=pl.initialise(G3,int(N*C_i))
        G4=pl.initialise(G4,int(N*C_i))

        #print("\t",j)
        
    R_random4[i]=np.mean(r1)
    R_lesmis4[i]=np.mean(r2)
    R_scalef4[i]=np.mean(r3)
    R_smallw4[i]=np.mean(r4)
    
    P_random4[i]=np.mean(p1)
    P_lesmis4[i]=np.mean(p2)
    P_scalef4[i]=np.mean(p3)
    P_smallw4[i]=np.mean(p4)
    
    F_random4[i]=np.mean(f1)
    F_lesmis4[i]=np.mean(f2)
    F_scalef4[i]=np.mean(f3)
    F_smallw4[i]=np.mean(f4)
    
    eR_random4[i]=np.std(r1)
    eR_lesmis4[i]=np.std(r2)
    eR_scalef4[i]=np.std(r3)
    eR_smallw4[i]=np.std(r4)
    
    eP_random4[i]=np.std(p1)
    eP_lesmis4[i]=np.std(p2)
    eP_scalef4[i]=np.std(p3)
    eP_smallw4[i]=np.std(p4)
    
    eF_random4[i]=np.std(f1)
    eF_lesmis4[i]=np.std(f2)
    eF_scalef4[i]=np.std(f3)
    eF_smallw4[i]=np.std(f4)
    
    print(i)
    
#%%
    

    
print("w_array=np.array([",np.array2string(w_array,separator=","),"])")
print(" ")
print(" ")


print("#RHPM")
print(" ")
print(" ")
    
      
print("#Steady State Resource")
print(" ")

print("R_random1=np.array(",np.array2string(R_random1,separator=","),")")
print(" ")
print("R_lesmis1=np.array(",np.array2string(R_lesmis1,separator=","),")")
print(" ")
print("R_scalef1=np.array(",np.array2string(R_scalef1,separator=","),")")
print(" ")
print("R_smallw1=np.array(",np.array2string(R_smallw1,separator=","),")")
print(" ")

print("eR_random1=np.array(",np.array2string(eR_random1,separator=","),")")
print(" ")
print("eR_lesmis1=np.array(",np.array2string(eR_lesmis1,separator=","),")")
print(" ")
print("eR_scalef1=np.array(",np.array2string(eR_scalef1,separator=","),")")
print(" ")
print("eR_smallw1=np.array(",np.array2string(eR_smallw1,separator=","),")")
print(" ")

print("#Total Profit")
print(" ")
print("P_random1=np.array(",np.array2string(P_random1,separator=","),")")
print(" ")
print("P_lesmis1=np.array(",np.array2string(P_lesmis1,separator=","),")")
print(" ")
print("P_scalef1=np.array(",np.array2string(P_scalef1,separator=","),")")
print(" ")
print("P_smallw1=np.array(",np.array2string(P_smallw1,separator=","),")")
print(" ")

print("eP_random1=np.array(",np.array2string(eP_random1,separator=","),")")
print(" ")
print("eP_lesmis1=np.array(",np.array2string(eP_lesmis1,separator=","),")")
print(" ")
print("eP_scalef1=np.array(",np.array2string(eP_scalef1,separator=","),")")
print(" ")
print("eP_smallw1=np.array(",np.array2string(eP_smallw1,separator=","),")")
print(" ")

print("#Frequency of Change")
print(" ")
print("F_random1=np.array(",np.array2string(F_random1,separator=","),")")
print(" ")
print("F_lesmis1=np.array(",np.array2string(F_lesmis1,separator=","),")")
print(" ")
print("F_scalef1=np.array(",np.array2string(F_scalef1,separator=","),")")
print(" ")
print("F_smallw1=np.array(",np.array2string(F_smallw1,separator=","),")")
print(" ")

print("eF_random1=np.array(",np.array2string(eF_random1,separator=","),")")
print(" ")
print("eF_lesmis1=np.array(",np.array2string(eF_lesmis1,separator=","),")")
print(" ")
print("eF_scalef1=np.array(",np.array2string(eF_scalef1,separator=","),")")
print(" ")
print("eF_smallw1=np.array(",np.array2string(eF_smallw1,separator=","),")")
print(" ")


print("#IHPM")
print(" ")
print(" ")
      
      
print("#Steady State Resource")
print(" ")
print("R_random2=np.array(",np.array2string(R_random2,separator=","),")")
print(" ")
print("R_lesmis2=np.array(",np.array2string(R_lesmis2,separator=","),")")
print(" ")
print("R_scalef2=np.array(",np.array2string(R_scalef2,separator=","),")")
print(" ")
print("R_smallw2=np.array(",np.array2string(R_smallw2,separator=","),")")
print(" ")

print("eR_random2=np.array(",np.array2string(eR_random2,separator=","),")")
print(" ")
print("eR_lesmis2=np.array(",np.array2string(eR_lesmis2,separator=","),")")
print(" ")
print("eR_scalef2=np.array(",np.array2string(eR_scalef2,separator=","),")")
print(" ")
print("eR_smallw2=np.array(",np.array2string(eR_smallw2,separator=","),")")
print(" ")

print("#Total Profit")
print(" ")
print("P_random2=np.array(",np.array2string(P_random2,separator=","),")")
print(" ")
print("P_lesmis2=np.array(",np.array2string(P_lesmis2,separator=","),")")
print(" ")
print("P_scalef2=np.array(",np.array2string(P_scalef2,separator=","),")")
print(" ")
print("P_smallw2=np.array(",np.array2string(P_smallw2,separator=","),")")
print(" ")

print("eP_random2=np.array(",np.array2string(eP_random2,separator=","),")")
print(" ")
print("eP_lesmis2=np.array(",np.array2string(eP_lesmis2,separator=","),")")
print(" ")
print("eP_scalef2=np.array(",np.array2string(eP_scalef2,separator=","),")")
print(" ")
print("eP_smallw2=np.array(",np.array2string(eP_smallw2,separator=","),")")
print(" ")

print("#Frequency of Change")
print(" ")
print("F_random2=np.array(",np.array2string(F_random2,separator=","),")")
print(" ")
print("F_lesmis2=np.array(",np.array2string(F_lesmis2,separator=","),")")
print(" ")
print("F_scalef2=np.array(",np.array2string(F_scalef2,separator=","),")")
print(" ")
print("F_smallw2=np.array(",np.array2string(F_smallw2,separator=","),")")
print(" ")

print("eF_random2=np.array([",np.array2string(eF_random2,separator=","),")")
print(" ")
print("eF_lesmis2=np.array([",np.array2string(eF_lesmis2,separator=","),")")
print(" ")
print("eF_scalef2=np.array([",np.array2string(eF_scalef2,separator=","),")")
print(" ")
print("eF_smallw2=np.array([",np.array2string(eF_smallw2,separator=","),")")
print(" ")


print("#GP")
print(" ")
print(" ")
      
      
print("#Steady State Resource")
print(" ")
print("R_random3=np.array(",np.array2string(R_random3,separator=","),")")
print(" ")
print("R_lesmis3=np.array(",np.array2string(R_lesmis3,separator=","),")")
print(" ")
print("R_scalef3=np.array(",np.array2string(R_scalef3,separator=","),")")
print(" ")
print("R_smallw3=np.array(",np.array2string(R_smallw3,separator=","),")")
print(" ")

print("eR_random3=np.array(",np.array2string(eR_random3,separator=","),")")
print(" ")
print("eR_lesmis3=np.array(",np.array2string(eR_lesmis3,separator=","),")")
print(" ")
print("eR_scalef3=np.array(",np.array2string(eR_scalef3,separator=","),")")
print(" ")
print("eR_smallw3=np.array(",np.array2string(eR_smallw3,separator=","),")")
print(" ")

print("#Total Profit")
print(" ")
print("P_random3=np.array(",np.array2string(P_random3,separator=","),")")
print(" ")
print("P_lesmis3=np.array(",np.array2string(P_lesmis3,separator=","),")")
print(" ")
print("P_scalef3=np.array(",np.array2string(P_scalef3,separator=","),")")
print(" ")
print("P_smallw3=np.array(",np.array2string(P_smallw3,separator=","),")")
print(" ")

print("eP_random3=np.array(",np.array2string(eP_random3,separator=","),")")
print(" ")
print("eP_lesmis3=np.array(",np.array2string(eP_lesmis3,separator=","),")")
print(" ")
print("eP_scalef3=np.array(",np.array2string(eP_scalef3,separator=","),")")
print(" ")
print("eP_smallw3=np.array(",np.array2string(eP_smallw3,separator=","),")")
print(" ")

print("#Frequency of Change")
print(" ")
print("F_random3=np.array(",np.array2string(F_random3,separator=","),")")
print(" ")
print("F_lesmis3=np.array(",np.array2string(F_lesmis3,separator=","),")")
print(" ")
print("F_scalef3=np.array(",np.array2string(F_scalef3,separator=","),")")
print(" ")
print("F_smallw3=np.array(",np.array2string(F_smallw3,separator=","),")")
print(" ")

print("eF_random3=np.array(",np.array2string(eF_random3,separator=","),")")
print(" ")
print("eF_lesmis3=np.array(",np.array2string(eF_lesmis3,separator=","),")")
print(" ")
print("eF_scalef3=np.array(",np.array2string(eF_scalef3,separator=","),")")
print(" ")
print("eF_smallw3=np.array(",np.array2string(eF_smallw3,separator=","),")")
print(" ")


print("#LP")
print(" ")
print(" ")
      
      
print("#Steady State Resource")
print(" ")
print("R_random4=np.array(",np.array2string(R_random4,separator=","),")")
print(" ")
print("R_lesmis4=np.array(",np.array2string(R_lesmis4,separator=","),")")
print(" ")
print("R_scalef4=np.array(",np.array2string(R_scalef4,separator=","),")")
print(" ")
print("R_smallw4=np.array(",np.array2string(R_smallw4,separator=","),")")
print(" ")

print("eR_random4=np.array(",np.array2string(eR_random4,separator=","),")")
print(" ")
print("eR_lesmis4=np.array(",np.array2string(eR_lesmis4,separator=","),")")
print(" ")
print("eR_scalef4=np.array(",np.array2string(eR_scalef4,separator=","),")")
print(" ")
print("eR_smallw4=np.array(",np.array2string(eR_smallw4,separator=","),")")
print(" ")

print("#Total Profit")
print(" ")
print("P_random4=np.array(",np.array2string(P_random4,separator=","),")")
print(" ")
print("P_lesmis4=np.array(",np.array2string(P_lesmis4,separator=","),")")
print(" ")
print("P_scalef4=np.array(",np.array2string(P_scalef4,separator=","),")")
print(" ")
print("P_smallw4=np.array(",np.array2string(P_smallw4,separator=","),")")
print(" ")

print("eP_random4=np.array(",np.array2string(eP_random4,separator=","),")")
print(" ")
print("eP_lesmis4=np.array(",np.array2string(eP_lesmis4,separator=","),")")
print(" ")
print("eP_scalef4=np.array(",np.array2string(eP_scalef4,separator=","),")")
print(" ")
print("eP_smallw4=np.array(",np.array2string(eP_smallw4,separator=","),")")
print(" ")

print("#Frequency of Change")
print(" ")
print("F_random4=np.array(",np.array2string(F_random4,separator=","),")")
print(" ")
print("F_lesmis4=np.array(",np.array2string(F_lesmis4,separator=","),")")
print(" ")
print("F_scalef4=np.array(",np.array2string(F_scalef4,separator=","),")")
print(" ")
print("F_smallw4=np.array(",np.array2string(F_smallw4,separator=","),")")
print(" ")

print("eF_random4=np.array(",np.array2string(eF_random4,separator=","),")")
print(" ")
print("eF_lesmis4=np.array(",np.array2string(eF_lesmis4,separator=","),")")
print(" ")
print("eF_scalef4=np.array(",np.array2string(eF_scalef4,separator=","),")")
print(" ")
print("eF_smallw4=np.array(",np.array2string(eF_smallw4,separator=","),")")
print(" ")

#%%

