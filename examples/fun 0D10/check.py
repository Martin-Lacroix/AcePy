import sys
sys.path.append('../../')
from fun0d10 import sampler,response
import numpy as np
import pickle

# %% Initialisation

f = open('model.pickle','rb')
model = pickle.load(f)
f.close()

nbrPts = int(1e5)
point = sampler(nbrPts)
resp = response(point)
respMod = model.eval(point)

# %% Monte Carlo and Error

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

error = abs(np.divide(resp-respMod,resp))
error = 100*np.mean(error,axis=0)