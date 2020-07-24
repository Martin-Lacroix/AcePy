import numpy as np
from matplotlib import pyplot as plt
from expo2d1 import sampler,response

# %% Monte Carlo

nbrPts = int(1e5)
point = sampler(nbrPts)
resp = response(point)
mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(mean.shape[0]): plt.plot(mean[i],'C0')
plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
for i in range(var.shape[0]): plt.plot(var[i],'C0')
plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Variance')
plt.legend()