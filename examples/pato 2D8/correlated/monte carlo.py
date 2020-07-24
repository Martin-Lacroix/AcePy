from matplotlib import pyplot as plt
import numpy as np

# %% Monte Carlo

nbrPts = int(2e4)
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(resp.shape[0],nbrPts,replace=0)
resp = resp[index]

mean = np.mean(resp,axis=0)
var = np.var(resp,axis=0)

np.save('mean.npy',mean)
np.save('var.npy',var)

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(3,11): plt.plot(mean[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel('Mean')

plt.figure(2)
for i in range(3,11): plt.plot(var[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel('Variance')

plt.figure(3)
plt.plot(mean[1],label='$\dot{m}_g$')
plt.ylabel('Mean')
plt.legend()

plt.figure(4)
plt.plot(var[1],label='$\dot{m}_g$')
plt.ylabel('Variance')
plt.legend()

plt.figure(5)
plt.plot(mean[0],label='Char')
plt.plot(mean[2],label='Virgin')
plt.ylabel('Mean')
plt.legend()

plt.figure(6)
plt.plot(var[0],label='Char')
plt.plot(var[2],label='Virgin')
plt.ylabel('Variance')
plt.legend()