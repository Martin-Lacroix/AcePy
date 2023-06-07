import numpy as np
import acepy as ap
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 20
ordQuad = 10*ordPoly
dist = ap.Normal(1,0.5)

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