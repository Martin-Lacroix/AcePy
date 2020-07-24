import sys
sys.path.append('../../')
from ishig0d3 import sampler,response
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

a = 7
b = 0.1

mean = a/2
var = a**2/8+b*np.pi**4/5+b**2*np.pi**8/18+1/2
S = np.array([0.5*(1+b*np.pi**4/5)**2,a**2/8,0])/var
St = np.array([0.5*(1+b*np.pi**4/5)**2+8*b**2*np.pi**8/225,a**2/8,8*b**2*np.pi**8/225])/var

meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)

error = abs(np.divide(resp-respMod,resp))
error = 100*np.mean(error,axis=0)