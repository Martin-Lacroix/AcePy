import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import numpy as np
import pickle

# %% Initialisation

f = open('model.pickle','rb')
model = pickle.load(f)
f.close()

nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
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

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18

plt.figure(1)
for i in range(3,11): plt.plot(meanMod[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.plot(mean[3:11].T,'--')
plt.ylabel('Mean')

plt.figure(2)
for i in range(3,11): plt.plot(varMod[i],label=str(i-2))
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.plot(var[3:11].T,'--')
plt.ylabel('Variance')

plt.figure(3)
plt.plot(meanMod[1],label='Chaoslib')
plt.plot(mean[1],'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.legend()

plt.figure(4)
plt.plot(varMod[1],'C0',label='Chaoslib')
plt.plot(var[1],'--C1',label='Monte carlo')
plt.ylabel('Variance')
plt.legend()

plt.figure(5)
plt.plot(meanMod[0],label='Chaoslib')
plt.plot(mean[0],'--',label='Monte Carlo')
plt.plot(meanMod[2],'C0',)
plt.plot(mean[2],'--C1')
plt.ylabel('Mean')
plt.legend()

plt.figure(6)
plt.plot(varMod[0],label='Chaoslib')
plt.plot(var[0],'--',label='Monte Carlo')
plt.plot(varMod[2],'C0')
plt.plot(var[2],'--C1')
plt.ylabel('Variance')
plt.legend()