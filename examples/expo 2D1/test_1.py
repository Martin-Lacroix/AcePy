import numpy as np
import acepy as ap
from fun import sampler,response
from matplotlib import pyplot as plt

# %% Initialisation

order = 30
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
index,weight = ap.fekquad(point,poly)

poly.trunc(15)
point = point[index]

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