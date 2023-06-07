import numpy as np
import acepy as ap
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 5
ordQuad = 2*ordPoly

dist = []
dist.append(ap.Normal(0.5,0.15))
dist.append(ap.Uniform(0.5,2.5))
dist.append(ap.Uniform(0.03,0.07))

# %% Polynomial Chaos

point,weight = ap.tensquad(ordQuad,dist)
poly = ap.polyrecur(ordPoly,dist)

resp = response(point)
coef = ap.spectral(resp,poly,point,weight)
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