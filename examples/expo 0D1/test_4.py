import numpy as np
import acepy as ap
from fun import response

# %% Initialisation

ordPoly = 7
ordQuad = 2*ordPoly
dist = ap.Normal(1,0.5)

# %% Polynomial Chaos

point,weight = ap.tensquad(ordQuad,dist)
resp = response(point)

poly = ap.polyrecur(ordPoly,dist)
coef = ap.spectral(resp,poly,point,weight)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')