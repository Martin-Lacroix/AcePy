import numpy as np
import acepy as ap
from fun import sampler
from fun import response

# %% Initialisation

order = 7
nbrPts = 1000
pdf = ap.Normal(1,0.5).pdf
dom = [-1,3]

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
resp = response(point)

coef,index = ap.lars(resp,poly,point,it=10)
coef = coef[index]
poly.clean(index)

model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')