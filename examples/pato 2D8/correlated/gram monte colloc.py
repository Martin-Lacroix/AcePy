import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 3
nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(3,11): plt.plot(mean[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.plot(meanMc[3:11].T,'--')
plt.ylabel('Mean')

plt.figure(2)
for i in range(3,11): plt.plot(var[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.plot(varMc[3:11].T,'--')
plt.ylabel('Variance')

plt.figure(3)
plt.plot(mean[1],label='Chaoslib')
plt.plot(meanMc[1],'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(4)
plt.plot(var[1],label='Chaoslib')
plt.plot(varMc[1],'--',label='Monte carlo')
plt.ylabel('Variance')
plt.legend()

plt.figure(5)
plt.plot(mean[0],label='Chaoslib')
plt.plot(meanMc[0],'--',label='Monte Carlo')
plt.plot(mean[2],'C0')
plt.plot(meanMc[2],'--C1')
plt.ylabel('Mean')
plt.legend()

plt.figure(6)
plt.plot(var[0],label='Chaoslib')
plt.plot(varMc[0],'--',label='Monte Carlo')
plt.plot(var[2],'C0')
plt.plot(varMc[2],'--C1')
plt.ylabel('Variance')
plt.legend()