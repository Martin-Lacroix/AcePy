from matplotlib import pyplot as plt
import numpy as np

# %% Monte Carlo

nbrPts = int(1e5)
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(resp.shape[0],nbrPts,replace=0)
resp = resp[index]

mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

# %% Save Results

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.plot(mean,label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
plt.plot(var,label='Monte Carlo')
plt.ylabel('Variance')
plt.legend()