import numpy as np
import chaoslib as cl
from fun import sampler
from fun import response

# %% Initialisation

order = 7
nbrPts = 1000
pdf = cl.Normal(1,0.5).pdf
dom = [-1,3]

# %% Polynomial Chaos

point = sampler(nbrPts)
resp = response(point)

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')