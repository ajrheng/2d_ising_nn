"""
Markov Chain Monte Carlo (MCMC) class to generate training samples for training the NN
"""

import numpy as np
import random

class ising2d_mcmc:

    def __init__(self,l,mc_steps=200000):

        self.l = l
        #self.num_samples = num_samples
        self.mc_steps = mc_steps

        random.seed() #set seed to current time

    def mcmc(self,t):
        """
        given the temperature t, return the state vector and average magnetization after running the MCMC sampling
        """
        
        self.state = np.ones((self.l,self.l),dtype=np.int8)
        num_measurements = 0
        m = 0
        
        for i in range(self.mc_steps):

            lx = random.randrange(self.l)
            ly = random.randrange(self.l)

            if self.state[lx,ly] == 1:
                change_s = -2
            else:
                change_s = 2

            n1 = self.left_neighbor_state(lx,ly)
            n2 = self.right_neighbor_state(lx,ly)
            n3 = self.top_neighbor_state(lx,ly)
            n4 = self.bottom_neighbor_state(lx,ly)

            Q = -1/t * (n1 + n2 + n3 + n4) * change_s

            if Q <= 0 or random.random() <= np.exp(-Q):
                self.state[lx,ly] = -self.state[lx,ly]

            if (i+1)%100 == 0:
                m += np.sum(self.state)
                num_measurements += 1

        m = m/(self.l**2 * num_measurements)
        return self.state, m

    def left_neighbor_state(self,x,y):
        if x == 0:
            return self.state[self.l-1,y]
        else:
            return self.state[x-1,y]
        
    def right_neighbor_state(self,x,y):
        if x == self.l-1:
            return self.state[0,y]
        else:
            return self.state[x+1,y]

            
    def top_neighbor_state(self,x,y):
        if y == self.l-1:
            return self.state[x,0]
        else:
            return self.state[x,y+1]
                
    def bottom_neighbor_state(self,x,y):
        if y == 0:
            return self.state[x,self.l-1]
        else:
            return self.state[x,y-1]
