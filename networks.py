#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:18:38 2021

@author: juliuslange
"""

import networkx as nx
import numpy as np
import player as pl
import random

def random_network(N,density):
    '''Create a Random Network
    Function to create a random network of given size and density
    
    
    Parameters
    __________
    
    N : int
        Number of Nodes
    density : float
        Float in range [0,1]. Fraction of all possible nodes occupied
        
    Returns
    _______
    
    G : networkx.network
        Random Network
        
    '''
    # calculate desired number of edges m
    E=N*(N-1)/2
    m=round(density*E)
    if m<1.5*N: return 0
    a=False
    while a==False:
        G=nx.gnm_random_graph(N,m)
        a = nx.is_connected(G)
    return G

def scale_free_network(N,m):
    '''Create a Scale-Free Network
    
    Function to create a Scale-Free network after the barabasi-Albert 
    algortithm. We start with a unconnected network of m nodes, then add (N-m)
    nodes one by one, each time adding m new edges which connect the new node
    to existing nodes according to preferential attachment

    Parameters
    __________
    
    N : int
        Number of Nodes
    m : int
        Number of new edges added per new node
        
    Returns
    _______
    
    G : networkx.network
        Scale-Free Network
    '''
    
    G=nx.barabasi_albert_graph(int(N),int(m))
   
    return G

def relaxed_caveman(N,l,p):
    k=int(N/l)
    G=nx.relaxed_caveman_graph(l,k,p)
    return G
 
def karate_club():
    G=nx.karate_club_graph()
    return G

def les_miserables():
    G=nx.les_miserables_graph()
    #mapping= {'Napoleon':0, 'Myriel':1, 'MlleBaptistine':2, 'MmeMagloire':3, 'CountessDeLo':4, 'Geborand':5, 'Champtercier':6, 'Cravatte':7, 'Count':8, 'OldMan':9, 'Valjean':10, 'Labarre':11, 'Marguerite':12, 'MmeDeR':13, 'Isabeau':14, 'Gervais':15, 'Listolier':16, 'Tholomyes':17, 'Fameuil':18, 'Blacheville':19, 'Favourite':20, 'Dahlia':21, 'Zephine':22, 'Fantine':23, 'MmeThenardier':24, 'Thenardier':25, 'Cosette':26, 'Javert':27, 'Fauchelevent':28, 'Bamatabois':29, 'Perpetue':30, 'Simplice':31, 'Scaufflaire':32, 'Woman1':33, 'Judge':34, 'Champmathieu':35, 'Brevet':36, 'Chenildieu':37, 'Cochepaille':38, 'Pontmercy':39, 'Boulatruelle':40, 'Eponine':41, 'Anzelma':42, 'Woman2':43, 'MotherInnocent':44, 'Gribier':45, 'MmeBurgon':46, 'Jondrette':47, 'Gavroche':48, 'Gillenormand':49,'Magnon':50, 'MlleGillenormand':51, 'MmePontmercy':52, 'MlleVaubois':53, 'LtGillenormand':54, 'Marius':55, 'BaronessT':56, 'Mabeuf':57, 'Enjolras':58, 'Combeferre':59, 'Prouvaire':60, 'Feuilly':61, 'Courfeyrac':62, 'Bahorel':63, 'Bossuet':64, 'Joly':65, 'Grantaire':66, 'MotherPlutarch':67, 'Gueulemer':68, 'Babet':69, 'Claquesous':70, 'Montparnasse':71, 'Toussaint':72, 'Child1':73, 'Child2':74, 'Brujon':75, 'MmeHucheloup':76}
    G=nx.convert_node_labels_to_integers(G, first_label=0)    
    #G = nx.relabel_nodes(G,mapping)
    return G

def nd_cube(d,l):
    dim=[l]*d
    G=nx.grid_graph(dim=dim)
    #mapping=dict(zip(G, range(0, l**d)))
    G=nx.convert_node_labels_to_integers(G, first_label=0)
    #G = nx.relabel_nodes(G,mapping)
    return G

def facebook_network():
    G = nx.read_edgelist("/Users/juliuslange/Downloads/facebook_combined.txt", create_using = nx.Graph(), nodetype=int)
    return G

def powerlaw_cluster(n,m,p):
    G=nx.powerlaw_cluster_graph(int(n),int(m),float(p))
    return G

def make_network(n,C_i,*argv):
    if n==0:
        G=random_network(argv[0],argv[1])
    elif n==1:
        G=scale_free_network(argv[0],argv[1])
    elif n==2:
        G=karate_club()
    elif n==3:
        G=les_miserables()
    elif n==6:
        G=facebook_network()
    elif n==7:
        G=powerlaw_cluster(argv[0],argv[1],argv[2])
    elif n==8:
        G=argv[0]
    elif n==9:
        G=nx.watts_strogatz_graph(argv[0],argv[1],argv[2])
    else: G=nx.complete_graph(argv[0])
    
    G=pl.initialise(G,C_i)
    return G

def remove_edges(G,N):
    for i in range(N):
        edges = np.array(G.edges())
        edges=edges.tolist()
        chosen_edge = random.choice(edges)    
        G.remove_edge(chosen_edge[0], chosen_edge[1])
    return G

def remove_cooperators(G):
    G_copy=G.copy()
    cooperators=[]
    for i in G.nodes:
        if G.nodes[i]["Player"].strategy==1:
            cooperators.append(i)
    G_copy.remove_nodes_from(cooperators)
    return G_copy

def remove_defectors(G):
    G_copy=G.copy()
    defectors=[]
    for i in G.nodes:
        if G.nodes[i]["Player"].strategy==0:
            defectors.append(i)
    G_copy.remove_nodes_from(defectors)
    return G_copy