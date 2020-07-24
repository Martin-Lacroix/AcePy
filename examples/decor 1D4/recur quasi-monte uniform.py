import sys
sys.path.append('../../')
from decor1d4 import response,norm
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Creation of the PCA whitening class

nbrPts = int(1e5)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = norm(point[index])
mapping = cl.Pca(point)

# Initialization

order = 4
nbrPts = 100
dom = [[-5,5]]*4
dist = cl.Joint([cl.Normal(0,1) for i in range(4)])

# %% Polynomial Chaos

point,weight = cl.qmcquad(nbrPts,dom,dist.pdf)
poly = cl.polyrecur(order,dist)
resp = response(mapping.corr(point))

coef = cl.colloc(resp,poly,point,weight)
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