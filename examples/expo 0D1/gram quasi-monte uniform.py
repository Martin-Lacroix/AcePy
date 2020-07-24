import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from expo0d1 import response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 15
nbrPts = 100
dist = cl.Normal(1,0.5)
dom = [-1,3]

# %% Polynomial Chaos

point,weight = cl.qmcquad(nbrPts,dom,dist.pdf)
resp = response(point)

poly = cl.gschmidt(order,point,weight)
coef = cl.colloc(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')