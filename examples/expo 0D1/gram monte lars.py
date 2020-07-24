import sys
sys.path.append('../../')
from expo0d1 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 7
nbrPts = 1000
pdf = cl.Normal(1,0.5).pdf
dom = [-1,3]

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
resp = response(point)

coef,index = cl.lars(resp,poly,point,it=10)
coef = coef[index]
poly.clean(index)

model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')