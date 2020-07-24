import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from expo2d1 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 20
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
index,weight = cl.simquad(point,poly)

poly.trunc(10)
point = point[index]

resp = response(point)
coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(mean.shape[0]):
    plt.plot(mean[i],'C0')
    plt.plot(meanMc[i],'C1--')

plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
for i in range(var.shape[0]):
    plt.plot(var[i],'C0')
    plt.plot(varMc[i],'C1--')

plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Variance')
plt.legend()