#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:41:43 2021

@author: juliuslange
"""

import networks as net
import networkx as nx
import numpy as np
import random

def continuous_mean_field(T,N,mc,e_l,e_h,R_i,C_i):
    def dR(R,x):
        dR=R*(1-R)-R*(x*e_l+(1-x)*e_h)
        if R<0:
            dR=0
        return dR

    def dx(R,x):
        #a=1/(0.1*N)
        #a=1/(77*0.0868)
        #a=1/(34*0.1390)
        a=x*(1-x)
        #a=1
        fitness_diff=(e_l*(R-mc)-e_h*(R-mc))
        dx=fitness_diff*a
        if x<0: dx=0
            
        return dx
    R=R_i
    x=C_i/N
    
    R_array=np.array([R_i])
    x_array=np.array([x])
    t_array=np.array([0])


    for i in range(T):
        x+=dx(R,x)
        R+=dR(R,x)
        R_array=np.append(R_array,R)
        x_array=np.append(x_array,x)        
        t_array=np.append(t_array,i+1)
        
    return t_array,R_array,x_array
    

def mean_field(T,N,mc,e_l,e_h,R_i,C_i):
    '''
    fuction to simulate the dynamics of a system using a discrete mean field
    approximation
    '''
    
    def dR(R,x):
        dR=R*(1-R)-R*(x*e_l+(1-x)*e_h)
        return dR

    def dx(R,x):
        #dx=(R/(N*N)*(e_l-e_h)+mc/(N*N)*(e_h-e_l))/d
        fitness_diff=(e_l*(R-mc)-e_h*(R-mc))

        #a=1/(N*0.1)
        #a=1/(77*0.0868)
        #a=1/(34*0.1390)
        #a=1/(N*d)
        #a=0.1
        #a=0.40
        #a=2*x*(1-x)*N/(N-1)
        a=4*x*(1-x)

        #a=1
        #a=x*(1-x)/(77*0.0868)
        #factor=N*(x-x*x)/(0.5*(N-1))
        factor=x*(1-x)
        #if fitness_diff<0: dx=a*factor*fitness_diff
        #elif fitness_diff>0: dx=a*factor*fitness_diff
        #else: dx=0
        dx=a*factor*fitness_diff
        #dx=d*(N-1)/(2)*x/N*(1-x)/N*(e_l*(R-mc)-e_h*(R-mc))
        '''
        chance=random.random()
        chance*=0.422
        if x*x*(1-x)*(1-x)>chance:
            if fitness_diff<0:
                return -1/N
            elif fitness_diff>0:
                return 1/N
            else: return 0
        else: return 0
        '''
        
        #if dx>1/(3*N): dx=1/N
        #elif dx<-1/(3*N): dx=-1/N
        if dx>1/N: dx=1/N
        elif dx<-1/N: dx=-1/N
        #else: dx=0
        return dx
        

            
    
    R=R_i
    x=C_i/N
    R_array=np.array([R_i])
    x_array=np.array([x])
    t_array=np.array([0])

    for i in range(T-1):
        x+=dx(R,x)
        #x=np.round(x*77)/77
        R+=dR(R,x)
        R_array=np.append(R_array,R)
        x_array=np.append(x_array,x)        
        t_array=np.append(t_array,i+1)
        
    return t_array,R_array,x_array

def dxAB(R,x_array,K,k_list,a,b,e_l,e_h,mc):
    '''
    fuction to calculate the change in the frequency of cooperators of nodes 
    of degree a because of nodes of degree b and their respective coupling
    '''
    if K[a,b]==0: return 0
    dxab=0
    fitness_diff=(e_l*(R-mc)-e_h*(R-mc))
    if fitness_diff>0:
        dxab=fitness_diff*K[a,b]*((1-x_array[a])*x_array[b])
    elif fitness_diff<0:
        dxab=fitness_diff*K[a,b]*(x_array[a]*(1-x_array[b]))
    elif fitness_diff==0:
        dxab=0
        
    #dxab=dxab*(x_array[a]*(1-x_array[a]))
    dxab=1*dxab/(k_list[a]*a)
    #if dxab<-1/k_list[a]: dxab=-1/k_list[a]
    #if dxab>1/k_list[a]: dxab=1/k_list[a]

    return dxab

def dx(R,x_array,k,K,k_list,e_l,e_h,mc,max_degree):
    '''
    fuction to calculate the change in the frequency of cooperators of nodes 
    of degree k because of all other node groups
    '''
    #dx=(R/(N*N)*(e_l-e_h)+mc/(N*N)*(e_h-e_l))/d
    if k_list[k]==0: return 0
    #f=x_array[a]
    dx=0
    for i in range(max_degree+1):
        dx+=dxAB(R,x_array,K,k_list,k,i,e_l,e_h,mc)
    
    #print(theta)

    #dx=d*(N-1)/(2)*x/N*(1-x)/N*(e_l*(R-mc)-e_h*(R-mc))
    #print(dx)
    #if dx<-1/k_list[k]: dx=-1/k_list[k]
    #if dx>1/k_list[k]: dx=1/k_list[k]
    return dx

def dR(R,x,e_l,e_h):
    '''
    Change in resource
    '''
    dR=R*(1-R)-R*(x*e_l+(1-x)*e_h)
    return dR


def heterogeneous_mean_field(T,G,mc,e_l,e_h,R_i,C_i,matrix,*argv):
    '''
    HMF simulation returning time series of resource and global frequency
    '''
   
    if matrix==True:
        K=argv[0]
    else: K=connectivity_matrix(G)
    #N=G.number_of_nodes()
    #M=G.size()
    #density=M/(0.5*N*(N-1))
    degree_list=np.array(G.degree)
    degree_list=degree_list[:,1]

    max_degree=np.amax(degree_list)
    
    k_list=degree_freq_list(G)

    #x_array=np.zeros([max_degree+1],dtype=float)
    #x_array[2]=1
    x_array=random_initialisation(G,C_i)

    R=R_i

    time=np.array([0])

    dx_array=np.zeros([max_degree+1],dtype=float)
    k_list=degree_freq_list(G)
    x=total_freq(x_array,k_list)
    print(K)
    resource=np.array([R_i])
    freq=np.array([x])
    #a=0.4
    #a=1
    for i in range(T-1):
        R_copy=R
        R+=dR(R,total_freq(x_array,k_list),e_l,e_h)
        
        for j in range(max_degree):
            dx_array[j]+=dx(R_copy,x_array,j,K,k_list,e_l,e_h,mc,max_degree)
            #if x_array[j]>1.01: x_array[j]=1
        #if x_array[j]<0: x_array[j]=0
        #print(x_array)

        x_array=np.add(x_array,dx_array)

        #print(x_array)
        #print(dx_array)
        #print(k_list)
        dx_array=np.zeros([max_degree+1],dtype=float)

        
        #X_avg_array=np.add(X_avg_array,x_array)
        time=np.append(time,i+1)
        resource=np.append(resource,R)
        freq=np.append(freq,total_freq(x_array,k_list))
        
    return(time,resource,freq)
    
def heterogeneous_mean_field_data(T,G,mc,e_l,e_h,R_i,C_i,matrix,*argv):
    '''
    HMF simulation returning an array of average frequency across 
    degree classes
    '''
    if matrix==True:
        K=argv[0]
    else: K=connectivity_matrix(G)
            
    degree_list=np.array(G.degree)
    degree_list=degree_list[:,1]

    max_degree=np.amax(degree_list)
    
    k_list=degree_freq_list(G)

    #x_array=np.zeros([max_degree+1],dtype=float)
    #x_array[2]=1
    x_array=random_initialisation(G,C_i)
    
    X_avg_array=np.zeros([max_degree+1],dtype=float)

    R=R_i
    
    time=np.array([0])

    k_list=degree_freq_list(G)
    #x=total_freq(x_array,k_list)
    #print(x)
    dx_array=np.zeros([max_degree+1],dtype=float)
    x=total_freq(x_array,k_list)
    #print(K)
    resource=np.array([R_i])
    freq=np.array([x])


    for i in range(T-1):
        R_copy=R
        R+=dR(R,total_freq(x_array,k_list),e_l,e_h)
        for j in range(max_degree):
            #dx_array[j]+=dx(R,x_array,j,K,k_list,e_l,e_h,mc,max_degree)
            dx_array[j]+=dx(R_copy,x_array,j,K,k_list,e_l,e_h,mc,max_degree)

            #if x_array[j]>1.01: x_array[j]=1
        #if x_array[j]<0: x_array[j]=0
        #print(x_array)

        x_array=np.add(x_array,dx_array)
    

        #print(x_array)
        #print(dx_array)
        #print(k_list)
        dx_array=np.zeros([max_degree+1],dtype=float)
        
        X_avg_array=np.add(X_avg_array,x_array)
        time=np.append(time,i)
        resource=np.append(resource,R)
        freq=np.append(freq,total_freq(x_array,k_list))
        
    X_avg_array/=T
        
    return k_list,X_avg_array,x_array
    
def total_freq(x_array,k_list):
    '''
    Funtion returning the global frequency for an array of degree frequencies
    '''
    x=0
    for i in range(x_array.size):
        x+=x_array[i]*k_list[i]
        #print(x)
#    return k_list,x/sum(k_list)
    return x/sum(k_list)

def degree_freq_list(G):

    N=G.number_of_nodes()
    degree_list=np.array(G.degree)
    degree_list=degree_list[:,1]
    max_degree=np.amax(degree_list)
    unique_degree_list=np.unique(degree_list)
    #print(unique_degree_list)
    #print(max_degree)
    deg_freq_list=np.zeros([max_degree+1],dtype=int)
    #print(deg_freq_list)
    for i in range(unique_degree_list.size):
        deg=unique_degree_list[i]
        for j in range(N):
            if G.degree[j]==deg:
                deg_freq_list[deg]+=1
                
    return deg_freq_list

def random_initialisation(G,C_i):
    '''
    Randomly initialise the degree frequency array with values which correspond
    to a given C_i
    '''
    degree_frequencies=degree_freq_list(G)
    degree_list=np.array(G.degree)
    degree_list=degree_list[:,1]
    max_degree=np.amax(degree_list)
    x_array=np.zeros([max_degree+1],dtype=float)
    N=G.number_of_nodes()
    node_list=np.zeros([N],dtype=int)
    for i in range(node_list.size):
        if i<C_i:
            node_list[i]=1
    np.random.shuffle(node_list)  
    for i in range(degree_frequencies.size):
        if degree_frequencies[i]==0: x_array[i]=0

        else:
            temp_list=node_list[:degree_frequencies[i]]
            node_list=node_list[degree_frequencies[i]:]

            x_array[i]=sum(temp_list)/degree_frequencies[i]
    #print(x_array)        
    return x_array

def connectivity_matrix(G):
    '''
    Function to determine the metrix containing the number of edges between 
    all different degree classes
    '''
    M=nx.adjacency_matrix(G)
    #N_edges=G.size()

    n_col=M.shape[0]
    n_row=M.shape[1]

    degree_list=np.array(G.degree)
    degree_list=degree_list[:,1]


    max_degree=np.amax(degree_list)

    K=np.zeros((max_degree+1,max_degree+1),dtype=float)

    for i in range(n_row):
        for j in range(n_col):
            element=M[i,j]
            #print(element)
            if element!=0:
                row_deg=np.array(degree_list[i])
                col_deg=np.array(degree_list[j])
                K[row_deg,col_deg]+=1
    print('matrix complete')           
    return K

    