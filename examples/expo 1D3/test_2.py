import numpy as np
import acepy as ap
from fun import sampler,response
from matplotlib import pyplot as plt

# %% Initialisation

order = 7
nbrPts = int(1e3)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
resp = response(point)

coef = ap.colloc(resp,poly,point)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')

plt.figure(1)
plt.plot(mean,label='AcePy')
plt.plot(meanMc,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.show()

plt.figure(2)
plt.plot(var,label='AcePy')
plt.plot(varMc,'--',label='Monte Carlo')
plt.ylabel('Variance')
plt.show()