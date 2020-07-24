from matplotlib import pyplot as plt
from fun0d10 import sampler,response
import seaborn as sns
import numpy as np

# %% Monte Carlo

nbrPts = int(1e6)
point = sampler(nbrPts)
resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.plot(point[:,0],point[:,2],'.C0')
plt.xlabel('$x_1$')
plt.ylabel('$x_3$')
