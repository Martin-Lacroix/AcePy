import pickle
import numpy as np
from fun import sampler,response
from matplotlib import pyplot as plt

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

plt.figure(1)
for i in range(mean.shape[0]):
    plt.plot(meanMod[i],'C0')
    plt.plot(mean[i],'C1--')

plt.legend(['AcePy','Monte Carlo'])
plt.ylabel('Mean')
plt.show()

plt.figure(2)
for i in range(var.shape[0]):
    plt.plot(varMod[i],'C0')
    plt.plot(var[i],'C1--')

plt.legend(['AcePy','Monte Carlo'])
plt.ylabel('Variance')
plt.show()