import numpy as np
import acepy as ap
from fun import sampler,response

# %% Initialisation

order = 12
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
index,weight = ap.simquad(point,poly)

poly.trunc(6)
point = point[index]
resp = response(point)

coef = ap.spectral(resp,poly,point,weight)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
sobol = ap.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = ap.ancova(model,point,weight)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')