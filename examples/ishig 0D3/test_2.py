import numpy as np
import chaoslib as cl
from fun import response

# %% Initialisation

order = 8
nbrPts = 1000
dom = ([-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi])
dist = cl.Joint([cl.Uniform(-np.pi,np.pi) for i in range(3)])

# %% Polynomial Chaos

point,weight = cl.qmcquad(nbrPts,dom,dist.pdf)
poly = cl.gschmidt(order,point,weight)
resp = response(point)

coef = cl.colloc(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
sobol = cl.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = cl.ancova(model,point,weight)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')