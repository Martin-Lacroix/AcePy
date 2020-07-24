import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Polynomial Chaos

nbrPtsTot = int(2e4)
order = np.array([1,2,3,4])
modelMonte = []
modelQuad = []

with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPtsTot,replace=0)
point = point[index]
resp = resp[index]

for i in range(order.shape[0]):
    
    # Sparse quadrature
    
    poly = cl.gschmidt(2*order[i],point)
    nbrPts = poly[:].shape[0]
    print(nbrPts)
    
    index,weight = cl.nulquad(point,poly)
    pointQ = point[index]
    respQ = resp[index]
    
    poly.trunc(order[i])
    coef = cl.spectral(respQ,poly,pointQ,weight)
    modelQuad.append(cl.Expansion(coef,poly))
    
    # Monte Carlo
    
    coef = cl.spectral(resp,poly,point)
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

index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

for i in range(len(order)):

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

# order = [1,2,3,4]
# mDotgQuad = [0.00141932,9.96895e-5,1.77638e-5,3.57326e-6]
# tempQuad = [4.24695e-5,7.54663e-6,2.04218e-6,5.47333e-7]
# charQuad = [0.306702,0.0869988,0.0341219,0.0255729]
# virgQuad = [0.000221771,6.9776e-5,5.40026e-5,4.3832e-5]
# mDotgMonte = [0.00110356,9.62695e-5,1.7647e-5,3.52985e-6]
# tempMonte = [3.20292e-5,7.43172e-6,2.01117e-6,5.40484e-7]
# charMonte = [0.227264,0.0802959,0.0339669,0.0257854]
# virgMonte = [0.000171873,6.49954e-5,5.2391e-5,4.35385e-5]

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
plt.semilogy(order,charMonte,'--C1',label="MC")
plt.semilogy(order,charMonte,'.C1',markersize=size)
plt.semilogy(order,charQuad,'--C0',label="Quad")
plt.semilogy(order,charQuad,'.C0',markersize=size)
plt.ylabel("Max Char SRE [-]")
plt.xlabel("Order [-]")
plt.legend()
plt.ylim([1e-2,1])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCharOrder.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.semilogy(order,virgMonte,'--C1',label="MC")
plt.semilogy(order,virgMonte,'.C1',markersize=size)
plt.semilogy(order,virgQuad,'--C0',label="Quad")
plt.semilogy(order,virgQuad,'.C0',markersize=size)
plt.ylabel("Max Virg SRE [-]")
plt.xlabel("Order [-]")
plt.ylim([1e-5,1e-3])
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoVirgOrder.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.semilogy(order,mDotgMonte,'--C1',label="MC")
plt.semilogy(order,mDotgMonte,'.C1',markersize=size)
plt.semilogy(order,mDotgQuad,'--C0',label="Quad")
plt.semilogy(order,mDotgQuad,'.C0',markersize=size)
plt.ylabel("Max $\dot{m}_g$ SRE [-]")
plt.xlabel("Order [-]")
plt.legend()
plt.ylim([1e-6,5e-3])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoMgOrder.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.semilogy(order,tempMonte,'--C1',label="MC")
plt.semilogy(order,tempMonte,'.C1',markersize=size)
plt.semilogy(order,tempQuad,'--C0',label="Quad")
plt.semilogy(order,tempQuad,'.C0',markersize=size)
plt.ylabel("Max $T$ SRE [-]")
plt.xlabel("Order [-]")
plt.legend()
plt.ylim([3e-7,1e-4])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoTOrder.pdf",bbox_inches="tight",format="pdf",transparent=True)  