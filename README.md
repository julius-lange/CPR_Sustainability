Code used in master project 'The Effects of Network Structure on the Sustainability of Common Pool Resources'

This is the code I used in all simulations for my master project. It is not super well documented, I know, but if there are any questions about what I did please don't hesitate to contact me!

The files contain the following code:

  player.py:    Basics of the stochasitc updating framework, player structure and updating mechnisms. Goes on to define functions to simulate the evolution of the                   system for various numbers of time-steps and compute appropriate statistics, lastly, some functions to compute the numbers of susceptible players.
  
  networks.py:  Definition of the basic networks and the initiation of any network through a general function. Functions to remove any kind of nodes which make it 
                easy to see any kind of structure of the arrangement of either the minoirty or the majority species.
                
  mean_field.py As it says. All mean field approaches used (continuous_mean_field()=MF, mean_field()=AMF, heterogeneous_mean_field()=HMF). Functions to make above                   calculations easier, e.g. to compute the connectivity matrix.
  
  Sustainability3.py:           File which conducts the simulation for my main results. Comptes the average steady-state resource level, total profit, and frequency                                 of change events for a variety of networks, updating procedures and marginal costs.
  
  degree_average_strategy.py:   Conducts the simulation used to obtain the degree dependence of the average strategy pursued by nodes of different degrees
  
  MF_corrections.py:             Simulates a number of systems to obatain the average numbers of susceptible nodes for different networks and marginal costs. Then                                    goes on to calculate the FA and SA model predictions.
  
