import numpy as np
import acepy as ap
from fun import sampler
from fun import response

# %% Initialisation

order = 2
nbrPts = int(1e5)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ap.gschmidt(order,point)
resp = response(point)

coef = ap.colloc(resp,poly,point)
model = ap.Expansion(coef,poly)

ap.save(model,'model')
mean,var = [model.mean,model.var]
index,ancova = ap.ancova(model,point)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')

# ----|------|------|------|
# X   | ST   | SS   | SC   |
# ----|------|------|------|
# X1  | 0    | 0    | 0    |
# X2  | 0    | 0    | 0    |
# X3  | 0.02 | 0.01 | 0.01 |
# X4  | 0.15 | 0.15 | 0    |
# X5  | 0.02 | 0.02 | 0    |
# X6  | 0.52 | 0.51 | 0.01 |
# X7  | 0.03 | 0.02 | 0.01 |
# X8  | 0    | 0    | 0    |
# X9  | 0    | 0    | 0    |
# X10 | 0    | 0    | 0    |
# X11 | 0    | 0    | 0    |
# X12 | 0.11 | 0.05 | 0.06 |
# X13 | 0.07 | 0.02 | 0.05 |
# X14 | 0.09 | 0.03 | 0.06 |
# -------------------------|