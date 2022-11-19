import numpy as np
import chaoslib as cl
from fun import sampler
from fun import response

# %% Initialisation

order = 4
nbrPts = int(1e5)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = cl.gschmidt(order,point)
resp = response(point)

coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

cl.save(model,'model')
sobol = cl.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = cl.ancova(model,point)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')

# -----------------------|
# ρ = 0 | ST  | SS  | SC |
# X1    | 0.2 | 0.2 | 0  |
# X2    | 0.6 | 0.6 | 0  |
# X1X2  | 0.2 | 0.2 | 0  |
# -----------------------------|
# ρ = 0.8 | ST   | SS   | SC   |
# X1      | 0.19 | 0.1  | 0.09 |
# X2      | 0.52 | 0.29 | 0.23 |
# X1X2    | 0.29 | 0.14 | 0.15 |
# -----------------------------|