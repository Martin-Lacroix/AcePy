import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Polynomial Chaos

nbrPtsTot = int(1e4)
order = np.array([3,4,5,6,7])
nbrMC = np.array([165,495,1287,3003,6435,10000])
modelMonte = []
modelQuad = []
nbrQuad = []

with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPtsTot,replace=0)
point = point[index]
resp = resp[index]

# Sparse quadrature

for i in range(order.shape[0]):
    
    poly = cl.gschmidt(order[i],point)
    nbrPts = poly[:].shape[0]
    nbrQuad.append(nbrPts)
    print(nbrPts)
    
    index,weight = cl.nulquad(point,poly)
    pointQ = point[index]
    respQ = resp[index]
    poly.trunc(3)
    
    coef = cl.spectral(respQ,poly,pointQ,weight)
    modelQuad.append(cl.Expansion(coef,poly))
    
# Monte Carlo

for i in range(nbrMC.shape[0]):
    
    index = np.random.choice(point.shape[0],nbrMC[i],replace=0)
    pointM = point[index]
    respM = resp[index]

    poly = cl.gschmidt(3,pointM)
    coef = cl.spectral(respM,poly,pointM)
    modelMonte.append(cl.Expansion(coef,poly))
    
# %% Error

nbrPts = int(2e4)
errorMonte = []
errorQuad = []
resp = []

with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

for i in range(len(nbrMC)):

    print("Error",i)
    
    respMod = modelQuad[i].eval(point)
    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    errorQuad.append(np.nanmax(np.divide(num,den),axis=1))
    
    respMod = modelMonte[i].eval(point)
    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    errorMonte.append(np.nanmax(np.divide(num,den),axis=1))
    
# %% Figures

errorQuad = np.array(errorQuad)
errorMonte = np.array(errorMonte)

mDotgMonte = errorMonte[:,1]
tempMonte = errorMonte[:,-1]
charMonte = errorMonte[:,0]
virgMonte = errorMonte[:,2]
mDotgQuad = errorQuad[:,1]
tempQuad = errorQuad[:,-1]
charQuad = errorQuad[:,0]
virgQuad = errorQuad[:,2]

# nbrQuad = [165,495,1287,3003,6435]
# nbrMC = [165,495,1287,3003,6435,10000]
# mDotgQuad = [0.00102228,2.63602e-5,1.70794e-5,1.55853e-5,1.5255e-5]
# tempQuad = [4.76936e-6,1.09233e-7,8.0382e-8,7.17605e-8,7.00229e-8]
# charQuad = [1.07546,0.26637,0.187786,0.162417,0.161303]
# virgQuad = [0.00116956,5.69422e-5,3.72502e-5,3.11777e-5,3.1654e-5]
# mDotgMonte = [404615,0.078453,0.000429555,2.66044e-5,1.5378e-5,1.52327e-5]
# tempMonte = [1535.75,0.000390353,2.57386e-6,1.49569e-7,7.0565e-08,6.98244e-8]
# charMonte = [9.01438e9,340.821,0.9089,0.203177,0.165702,0.15999]
# virgMonte = [1.11765e6,0.217949,0.000848279,0.000105097,3.29099e-05,3.0767e-5]

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.figure(1)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.loglog(nbrMC,charMonte,'--C1',label="MC")
plt.loglog(nbrMC,charMonte,'.C1',markersize=size)
plt.loglog(nbrQuad,charQuad,'--C0',label="Quad")
plt.loglog(nbrQuad,charQuad,'.C0',markersize=size)
plt.ylabel("Max Char SRE [-]")
plt.xlabel("Points [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharPoint.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.loglog(nbrMC,virgMonte,'--C1',label="MC")
plt.loglog(nbrMC,virgMonte,'.C1',markersize=size)
plt.loglog(nbrQuad,virgQuad,'--C0',label="Quad")
plt.loglog(nbrQuad,virgQuad,'.C0',markersize=size)
plt.ylabel("Max Virg SRE [-]")
plt.xlabel("Points [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoVirgPoint.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.loglog(nbrMC,mDotgMonte,'--C1',label="MC")
plt.loglog(nbrMC,mDotgMonte,'.C1',markersize=size)
plt.loglog(nbrQuad,mDotgQuad,'--C0',label="Quad")
plt.loglog(nbrQuad,mDotgQuad,'.C0',markersize=size)
plt.ylabel("Max $\dot{m}_g$ SRE [-]")
plt.xlabel("Points [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMgPoint.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.loglog(nbrMC,tempMonte,'--C1',label="MC")
plt.loglog(nbrMC,tempMonte,'.C1',markersize=size)
plt.loglog(nbrQuad,tempQuad,'--C0',label="Quad")
plt.loglog(nbrQuad,tempQuad,'.C0',markersize=size)
plt.ylabel("Max $T$ SRE [-]")
plt.xlabel("Points [-]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoTPoint.pdf",bbox_inches="tight",format="pdf",transparent=True)