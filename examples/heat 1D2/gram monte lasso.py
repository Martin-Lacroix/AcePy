import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from heat1d2 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 7
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
resp = response(point)

coef,index = cl.lasso(resp,poly,point,it=10)
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