import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from expo0d1 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 30
nbrPts = int(1e5)
dist = cl.Normal(1,0.5)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
index,weight = cl.fekquad(point,poly)

poly.trunc(15)
point = point[index]

resp = response(point)
coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')