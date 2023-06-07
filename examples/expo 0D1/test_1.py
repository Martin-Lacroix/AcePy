import numpy as np
import acepy as ap
from fun import sampler
from fun import response

# %% Initialisation

order = 30
nbrPts = int(1e5)
dist = ap.Normal(1,0.5)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
index,weight = ap.fekquad(point,poly)

poly.trunc(15)
point = point[index]

resp = response(point)
coef = ap.spectral(resp,poly,point,weight)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')