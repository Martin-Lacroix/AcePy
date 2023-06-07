import numpy as np
import acepy as ap
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 7
ordQuad = 3*ordPoly

dist = []
dist.append(ap.Normal(-18,2))
dist.append(ap.Gamma(2,0.001))

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
for i in range(mean.shape[0]):
    plt.plot(mean[i],'C0')
    plt.plot(meanMc[i],'C1--')

plt.legend(['AcePy','Monte Carlo'])
plt.ylabel('Mean')
plt.show()

plt.figure(2)
for i in range(var.shape[0]):
    plt.plot(var[i],'C0')
    plt.plot(varMc[i],'C1--')

plt.legend(['AcePy','Monte Carlo'])
plt.ylabel('Variance')
plt.show()