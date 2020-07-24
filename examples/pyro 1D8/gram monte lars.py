import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 3
nbrPts = int(1e4)

with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point)
coef,index = cl.lars(resp,poly,point,it=20)
coef = coef[index]
poly.clean(index)

model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
plt.plot(mean,label='Chaoslib')
plt.plot(meanMc,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
plt.plot(var,label='Chaoslib')
plt.plot(varMc,'--',label='Monte Carlo')
plt.ylabel('Variance')
plt.legend()