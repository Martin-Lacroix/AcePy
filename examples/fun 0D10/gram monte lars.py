import sys
sys.path.append('../../')
from fun0d10 import sampler,response
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 3
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
resp = response(point)

coef,index = cl.lars(resp,poly,point,it=80)
coef = coef[index]
poly.clean(index)

coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')