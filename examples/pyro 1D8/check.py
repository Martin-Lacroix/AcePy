import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
import numpy as np
import pickle

# %% Initialisation

f = open('model.pickle','rb')
model = pickle.load(f)
f.close()

nbrPts = int(1e5)
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

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
plt.plot(meanMod,label='Chaoslib')
plt.plot(mean,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.xlabel('Step')
plt.legend()

plt.figure(2)
plt.plot(varMod,label='Chaoslib')
plt.plot(var,'--',label='Monte Carlo')
plt.ylabel('Variance')
plt.legend()