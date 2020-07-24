import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 6
nbrPts = int(1e3)
with np.load('pointALL.npz',mmap_mode='r') as file: point = file['pts']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

# %% Fekete points

ptsAE = point[:,5:7]
poly = cl.gschmidt(order,ptsAE)
index,weight = cl.fekquad(ptsAE,poly)
fekAE = ptsAE[index]

ptsFN = point[:,[4,7]]
poly = cl.gschmidt(order,ptsFN)
index,weight = cl.fekquad(ptsFN,poly)
fekFN = ptsFN[index]

# %% Figures

width = 0.5
grid = [0.9,0.9,0.9]

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.plot(ptsAE[:,0],ptsAE[:,1],'.',label="MC")
plt.plot(fekAE[:,0],fekAE[:,1],'.',markersize=12,label="Fekete")
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.grid(linewidth=width,color=grid)
plt.legend()
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("fekAE.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.plot(ptsFN[:,0],ptsFN[:,1],'.')
plt.plot(fekFN[:,0],fekFN[:,1],'.',markersize=12)
plt.xlabel("$F_2$ [-]")
plt.ylabel("$n_2$ [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("fekFN.pdf",bbox_inches="tight",format="pdf",transparent=True)