import numpy as np
import chaoslib as cl
from fun import response

# %% Initialisation

ordPoly = 7
ordQuad = 2*ordPoly
dist = cl.Normal(1,0.5)

# %% Polynomial Chaos

point,weight = cl.tensquad(ordQuad,dist)
resp = response(point)

poly = cl.polyrecur(ordPoly,dist)
coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')