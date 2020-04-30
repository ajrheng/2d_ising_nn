"""
Plot the paramagnetic and ferromagnetic phases.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/alvin/Desktop/2d_ising_nn/data')

from ising2d_mcmc import *

if __name__ == "__main__":
    
    # Critical Tc = 2/log(1+sqrt(2)) ~ 2.3
    # Ferromagnetic phase first, t = 0.1

    t = 0.1
    l = 16
    mcmc_obj = ising2d_mcmc(l)

    ferro_state , ferro_mag = mcmc_obj.mcmc(t)
    ferro_state = ferro_state.reshape((l,l))

    print("average magnetization in the ferromagnetic phase = ", ferro_mag)

    f=plt.figure()
    plt.imshow(ferro_state,vmin=-1,vmax=1,origin='lower')
    plt.title('ferromagnetic')
    plt.axis('off')
    plt.draw()
    f.savefig('ferromagnet.png',dpi=300)

    # Paramagnetic phase next, t = 3

    t = 3
    mcmc_obj = ising2d_mcmc(l)

    para_state , para_mag = mcmc_obj.mcmc(t)
    para_state = para_state.reshape((l,l))

    print("average magnetization in the paramagnetic phase = ", para_mag)

    f=plt.figure()
    plt.imshow(para_state,vmin=-1,vmax=1,origin='lower')
    plt.title('paramagnetic')
    plt.axis('off')
    plt.draw()
    f.savefig('paramagnet.png',dpi=300)

    plt.show()