from matplotlib import pyplot as plt
from pyro1d4 import response
import numpy as np

# %% Monte Carlo

nbrPts = int(1e5)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

# %% Save Results

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
plt.plot(mean,label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
plt.plot(var,label='Monte Carlo')
plt.ylabel('Variance')
plt.legend()