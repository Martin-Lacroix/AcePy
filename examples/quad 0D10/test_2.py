import numpy as np
import acepy as ap
from fun import sampler,response

# %% Initialisation

order = 3
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
resp = response(point)

coef,index = ap.lars(resp,poly,point,it=80)
coef = coef[index]
poly.clean(index)

coef = ap.colloc(resp,poly,point)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')