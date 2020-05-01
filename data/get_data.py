"""
Driver class to run the MCMC and get the training results

Code is parallelized with joblib. It will use all available threads on your PC or all threads requested in your cluster job.
"""
#from tqdm import tqdm
import numpy as np
import random
from joblib import Parallel, delayed

from ising2d_mcmc import *

def one_hot_encoding(x):
    arr = np.zeros(2,dtype=np.int8)
    arr[x] = 1
    return arr

def run(t):

    mcmc_obj = ising2d_mcmc(l)
    state, mag = mcmc_obj.mcmc(t)

    return state.flatten(), mag


if __name__ == "__main__":

    random.seed()
    num_samples = 50
    tc = 2/np.log(1+np.sqrt(2))
    l = 16
    dataset = np.zeros((num_samples,l*l),dtype=np.int8)
    results = np.zeros((num_samples,2),dtype=np.int8)
    temp = np.array([random.uniform(0.1,4.0) for _ in range(num_samples)],dtype=np.float64)
    magnetization = np.zeros(num_samples,dtype=np.float64)

    output = Parallel(n_jobs=-1,verbose=8)(delayed(run)(t) for t in temp)

    # for i in tqdm(range(num_samples)):

    #     mcmc_obj = ising2d_mcmc(l)
    #     tval = random.uniform(0.1,4.0)
    #     temp[i] = tval
    #     state, magnetization = mcmc_obj.mcmc(tval)

    for i in range(len(output)):
        dataset[i,:] = output[i][0]
        results[i,:] = one_hot_encoding(int(temp[i] < tc)) #first index of OHE is 1 for paramagnet, second index is 1 for ferromagnet
        magnetization[i] = output[i][1]

    np.savez_compressed('data',a=dataset, b=results, c=temp, d = magnetization)
