import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np
import copy

# %% Initialisation

nbrPts = int(2e4)
with np.load('point.npz',mmap_mode='r') as file: point = file['pts']
with np.load('resp.npz',mmap_mode='r') as file: resp = file['resp']
index = np.random.choice(point.shape[0],nbrPts,replace=0)
poinTot = point[index]
respTot = resp[index]

# %% Monte Carlo

order = 3
modMc = []
ptList = nbrMC = np.array([165,495,1287,3003,6435,1e4,2e4]).astype(int)

for i in range(len(ptList)):
    
    index = np.random.choice(poinTot.shape[0],ptList[i],replace=0)
    point = poinTot[index]
    resp = respTot[index]
    
    poly = cl.gschmidt(order,point)
    coef = cl.spectral(resp,poly,point)
    model = cl.Expansion(coef,poly)
    modMc.append(model)
    
# %% Quadrature

order = 3
modQuad = []
quadList = []
nbrPts = int(2e4)
ordList = [7,6,5,4,3]

index = np.random.choice(poinTot.shape[0],nbrPts,replace=0)
point = poinTot[index]
resp = respTot[index]

polyTot = cl.gschmidt(ordList[0],point)
weight = 0

for i in range(len(ordList)):

    poly = copy.deepcopy(polyTot)
    poly.trunc(ordList[i])
    quadList.append(poly[:].shape[0])
    print(quadList[i])
    
    index,weight = cl.newquad(point,poly,weight)
    
    poly.trunc(order)
    point = point[index]
    resp = resp[index]
    
    coef = cl.spectral(resp,poly,point,weight)
    
    model = cl.Expansion(coef,poly)
    modQuad.append(model)

# %% Error

point = poinTot
resp = respTot
errQuad = []
errMc = []

for i in range(len(ptList)):
    
    print('Error - ',i)
    respMc = modMc[i].eval(point)
    num = np.mean((resp-respMc)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    err = np.nanmax(np.divide(num,den),axis=1)
    errMc.append(err)
    
for i in range(len(quadList)):
    
    print('Error - ',i)
    respQuad = modQuad[i].eval(point)
    num = np.mean((resp-respQuad)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    err = np.nanmax(np.divide(num,den),axis=1)
    errQuad.append(err)

errMc = np.array(errMc)
errQuad = np.array(errQuad)
mDotgQuad = errQuad[:,1]
virgQuad = errQuad[:,2]
mDotgMc = errMc[:,1]
virgMc = errMc[:,2]

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

index = 1
grid = [0.9,0.9,0.9]
width = 0.5
size = 8

# quadList = [6435,3003,1287,495,165]
# nbrMC = [165,495,1287,3003,6435,10000,20000]
# mDotgQuad = [1.49297e-05,1.51732e-05,1.67381e-05,2.33179e-05,0.000791672]
# virgQuad = [2.97631e-05,3.01217e-05,3.37883e-05,4.66131e-05,0.000989149]
# mDotgMonte = [5601.06,0.0542761,0.000295751,1.77739e-05,1.63915e-05,1.50728e-05,1.49067e-05]
# virgMonte = [11229.8,0.105034,0.000919611,5.81597e-05,3.587e-05,3.08477e-05,2.97007e-05]

plt.figure(1)
plt.loglog(ptList,mDotgMc,'--C0',label='Monte Carlo')
plt.loglog(ptList,mDotgMc,'.C0',markersize=size)
plt.semilogy(quadList,mDotgQuad,'--C3',label='Quadrature')
plt.semilogy(quadList,mDotgQuad,'.C3',markersize=size)
plt.ylabel('$\dot{m}_g$ SRE [-]')
plt.xlabel('Points [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("mDotgQuadMc.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.loglog(ptList,virgMc,'--C0',label='Monte Carlo')
plt.loglog(ptList,virgMc,'.C0',markersize=size)
plt.semilogy(quadList,virgQuad,'--C3',label='Quadrature')
plt.semilogy(quadList,virgQuad,'.C3',markersize=size)
plt.ylabel('Virg $d$ SRE [-]')
plt.xlabel('Points [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("virgQuadMc.pdf",bbox_inches="tight",format="pdf",transparent=True)