import sys
sys.path.append('../../')
from ishig0d3 import response
import chaoslib as cl
import numpy as np

# %% Initialisation

ordPoly = 8
ordQuad = 2*ordPoly
dist = [cl.Uniform(-np.pi,np.pi) for i in range(3)]

# %% Polynomial Chaos

point,weight = cl.tensquad(ordQuad,dist)
poly = cl.polyrecur(ordPoly,dist)

resp = response(point)
coef,index = cl.lars(resp,poly,point,it=20)
poly.clean(index)
coef = coef[index]

model = cl.Expansion(coef,poly)

cl.save(model,'model')
sobol = cl.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = cl.ancova(model,point,weight)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')