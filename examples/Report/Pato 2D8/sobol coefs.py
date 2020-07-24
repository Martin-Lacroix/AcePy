import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import seaborn as sns
import numpy as np

# %% Distribution

dist = []

dist.append(cl.Uniform(0.1,0.4))
dist.append(cl.Normal(1.2e4,1e3))
dist.append(cl.Normal(7.1e4,4e3))
dist.append(cl.Normal(3,0.2))

dist.append(cl.Uniform(0.1,0.3))
dist.append(cl.Normal(5e8,2e7))
dist.append(cl.Normal(1.7e5,6e3))
dist.append(cl.Normal(3,0.2))

dist = cl.Joint(dist)

# %% Initialisation

order = 3
nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point)
coef = cl.colloc(resp,poly,point)
model = cl.Expansion(coef,poly)

sobol = cl.anova(coef,poly)
S = np.mean(sobol['S'][:,:,1:],axis=2)
ST = np.mean(sobol['ST'][:,:,1:],axis=2)

# %% Figures

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'
