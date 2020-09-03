import numpy as np
import chaoslib as cl
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

order = 20
nbrPts = 200
dist = cl.Normal(1,0.5)
dom = [-1,3]

# %% Polynomial Chaos

point,weight = cl.qmcquad(nbrPts,dom,dist.pdf)
poly = cl.gschmidt(order,point,weight)
resp = response(point)

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