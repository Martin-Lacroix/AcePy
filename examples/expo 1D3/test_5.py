from sys import path
path.append('/mnt/Data/Devs/PCE-Chaoslib')

import numpy as np
import chaoslib as cl
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 5
ordQuad = 2*ordPoly

dist = []
dist.append(cl.Normal(0.5,0.15))
dist.append(cl.Uniform(0.5,2.5))
dist.append(cl.Uniform(0.03,0.07))

# %% Polynomial Chaos

point,weight = cl.tensquad(ordQuad,dist)
poly = cl.polyrecur(ordPoly,dist)

resp = response(point)
coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')

plt.figure(1)
plt.plot(mean,label='Chaoslib')
plt.plot(meanMc,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.show()

plt.figure(2)
plt.plot(var,label='Chaoslib')
plt.plot(varMc,'--',label='Monte Carlo')
plt.ylabel('Variance')
plt.show()