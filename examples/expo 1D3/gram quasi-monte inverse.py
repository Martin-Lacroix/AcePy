import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from expo1d3 import response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 5
nbrPts = 100
dist = cl.Joint([cl.Normal(0.5,0.15),cl.Uniform(0.5,2.5),cl.Uniform(0.03,0.07)])

# %% Polynomial Chaos

point = dist.halton(nbrPts)
poly = cl.gschmidt(order,point)
resp = response(point)

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
plt.plot(mean,label='Chaoslib')
plt.plot(meanMc,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
plt.plot(var,'C0',label='Chaoslib')
plt.plot(varMc,'C1--',label='Monte Carlo')
plt.ylabel('Variance')
plt.legend()