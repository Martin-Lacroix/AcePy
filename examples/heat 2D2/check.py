import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from heat2d2 import sampler,response
import numpy as np
import pickle

# %% Initialisation

f = open('model.pickle','rb')
model = pickle.load(f)
f.close()

nbrPts = int(1e4)
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

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(mean.shape[0]):
    plt.plot(meanMod[i],'C0')
    plt.plot(mean[i],'C1--')

plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Mean')
plt.legend()

plt.figure(2)
for i in range(var.shape[0]):
    plt.plot(varMod[i],'C0')
    plt.plot(var[i],'C1--')

plt.legend(['Chaoslib','Monte Carlo'])
plt.ylabel('Variance')
plt.legend()