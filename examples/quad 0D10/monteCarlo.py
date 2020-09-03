import numpy as np
from fun import sampler,response

# %% Monte Carlo

nbrPts = int(1e6)
point = sampler(nbrPts)
resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)
