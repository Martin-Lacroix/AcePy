import sys
sys.path.append('../../')
from ishig0d3 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 12
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
index,weight = cl.simquad(point,poly)

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