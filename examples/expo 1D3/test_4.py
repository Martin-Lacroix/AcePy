import numpy as np
import chaoslib as cl
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

order = 5
nbrPts = 100
dom = ([0,1],[0.5,2.5],[0.03,0.07])
dist = cl.Joint([cl.Normal(0.5,0.15),cl.Uniform(0.5,2.5),cl.Uniform(0.03,0.07)])

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