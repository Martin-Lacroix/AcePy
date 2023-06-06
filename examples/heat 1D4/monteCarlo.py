import numpy as np
from fun import sampler,response
from matplotlib import pyplot as plt

# %% Monte Carlo

nbrPts = int(2e5)
point = sampler(nbrPts)
resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

plt.figure(1)
plt.plot(mean,label='Monte-Carlo')
plt.ylabel('Mean')
plt.show()

plt.figure(2)
plt.plot(var,label='Monte-Carlo')
plt.ylabel('Variance')
plt.show()