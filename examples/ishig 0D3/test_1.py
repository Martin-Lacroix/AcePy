import numpy as np
import chaoslib as cl
from fun import sampler,response

# %% Initialisation

order = 12
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
index,weight = cl.fekquad(point,poly)

poly.trunc(6)
point = point[index]
resp = response(point)

coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
sobol = cl.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = cl.ancova(model,point,weight)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')