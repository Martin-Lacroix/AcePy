import sys
sys.path.append('../../')
from expo0d1 import response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 15
nbrPts = 100
dist = cl.Normal(1,0.5)

# %% Polynomial Chaos

point = dist.halton(nbrPts)
resp = response(point)

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')