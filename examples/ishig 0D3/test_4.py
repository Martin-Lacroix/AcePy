import numpy as np
import acepy as ap
from fun import response

# %% Initialisation

ordPoly = 8
ordQuad = 2*ordPoly
dist = [ap.Uniform(-np.pi,np.pi) for i in range(3)]

# %% Polynomial Chaos

point,weight = ap.tensquad(ordQuad,dist)
poly = ap.polyrecur(ordPoly,dist)

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