# Computing Very Large Correlation Matrices in Parallel

import os
# compute in parallel
from multiprocessing import Pool
import numpy as np
import pandas as pd
import deepgraph as dg

'''Create a set variables and store it as a 2d-matrix X (shape=(n_samples, n_features)) on disc. 
To speed up the computation of the Pearson correlation coefficients later on, we whiten each variable.
'''
# create observations
from numpy.random import RandomState
prng = RandomState(0)
n_samples = int(1e2)
n_features = int(1e1)
X = prng.randint(100, size=(n_samples, n_features)).astype(np.float64)

Xvar = X.var(axis=1, keepdims=True, ddof=1)
# whiten variables for fast parallel computation later on
X = X - X.mean(axis=1, keepdims=True)
#X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# save in binary format
np.save('samples', X)


# parameters (change these to control RAM usage)
step_size = 1e5
n_processes = 100

# load samples as memory-map
X = np.load('samples.npy', mmap_mode='r')

# create node table that stores references to the mem-mapped samples
v = pd.DataFrame({'index': range(X.shape[0])})

# connector function to compute pairwise pearson correlations
def corr(index_s, index_t):
    samples_s = X[index_s]
    samples_t = X[index_t]
    corr = np.einsum('ij,ij->i', samples_s, samples_t) / (n_features-1)
    return corr

# index array for parallelization
pos_array = np.array(np.linspace(0, n_samples*(n_samples-1)//2, n_processes), dtype=int)

# parallel computation
def create_ei(i):

    from_pos = pos_array[i]
    to_pos = pos_array[i+1]

    # initiate DeepGraph
    g = dg.DeepGraph(v)

    # create edges
    g.create_edges(connectors=corr, step_size=step_size, 
                   from_pos=from_pos, to_pos=to_pos)

    # store edge table
    g.e.to_pickle('/tmp/corr/{}.pickle'.format(str(i).zfill(3)))

# computation
if __name__ == '__main__':
    indices = np.arange(0, n_processes - 1)
    p = Pool()
    for _ in p.imap_unordered(create_ei, indices):
        pass



'''Collect the computed correlation values and store them in an hdf file.
'''
# store correlation values
files = os.listdir('/tmp/corr/')
files.sort()
store = pd.HDFStore('e.h5', mode='w')
for f in files:
    et = pd.read_pickle('/tmp/corr/{}'.format(f))
    store.append('e', et, format='t', data_columns=True, index=False)
store.close()
