import numpy as np
import chaoslib as cl
from fun import sampler,response
from matplotlib import pyplot as plt

# %% Initialisation

order = 15
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
index,weight = cl.newquad(point,poly)

poly.trunc(7)
point = point[index]
resp = response(point)

coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')

plt.figure(1)
plt.plot(mean,'C0',label='Chaoslib')
plt.plot(meanMc,'C1--',label='Monte Carlo')
plt.ylabel('Mean')
plt.show()

plt.figure(2)
plt.plot(var,'C0',label='Chaoslib')
plt.plot(varMc,'C1--',label='Monte Carlo')
plt.ylabel('Variance')
plt.show()