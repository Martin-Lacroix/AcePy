import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np
import parula

# %% Initialisation

order = 7
nbrPts = int(1e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

# %% Polynomial Chaos

poly = cl.gschmidt(order,point,trunc=0.85)
index,weight = cl.nulquad(point,poly)
print(weight.shape[0])

poly.trunc(3)
point = point[index]
resp = resp[index]

coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

# %% Error

char = resp[:,0,:]
charMod = model.eval(point)[:,0,:]

den = char**2
num = (char-charMod)**2
error = np.nanmax(np.divide(num,den),axis=1)
errorLog = np.log(error/np.min(error))

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'
parula = parula.colour()

grid = [0.9,0.9,0.9]
width = 0.5

plt.figure(1,figsize=(5,4))
fig = plt.scatter(point[:,3],point[:,7],c=errorLog,cmap=parula)
plt.xlabel("$n_1$ [-]")
plt.ylabel("$n_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharQuadN.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2,figsize=(6.5,4))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
fig = plt.scatter(point[:,2],point[:,6],c=errorLog,cmap=parula)
cbar = plt.colorbar(fig,aspect=10)
ticksLog = np.array([0,2,4,6,8,10])
ticks = np.exp(ticksLog)*np.min(error)
ticks = np.round(ticks,2)
cbar.set_ticks(ticksLog)
cbar.set_ticklabels(ticks)
cbar.set_label('Max Char SRE [-]',labelpad=10)
plt.xlabel("$E_1$ [J/mol]")
plt.ylabel("$E_2$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharQuadE.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(5,4))
fig = plt.scatter(point[:,0],point[:,4],c=errorLog,cmap=parula)
plt.xlabel("$F_1$ [-]")
plt.ylabel("$F_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharQuadF.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4,figsize=(6.5,4))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
fig = plt.scatter(point[:,1],point[:,5],c=errorLog,cmap=parula)
cbar = plt.colorbar(fig,aspect=10)
ticksLog = np.array([0,2,4,6,8,10])
ticks = np.exp(ticksLog)*np.min(error)
ticks = np.round(ticks,2)
cbar.set_ticks(ticksLog)
cbar.set_ticklabels(ticks)
cbar.set_label('Max Char SRE [-]',labelpad=10)
plt.xlabel("$A_1$ [1/s]")
plt.ylabel("$A_2$ [1/s]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharQuadA.pdf",bbox_inches="tight",format="pdf",transparent=True)