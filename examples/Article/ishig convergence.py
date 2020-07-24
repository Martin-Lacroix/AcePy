import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
from ishig0d3 import sampler,response
import chaoslib as cl
import numpy as np
import copy

# %% Monte Carlo

order = 9
modMc = []
ptList = np.array([10**2.4,10**2.6,1e3,10**3.5,1e4,10**4.5]).astype(int)

for i in range(len(ptList)):
    
    point = sampler(ptList[i])
    poly = cl.gschmidt(order,point)
    
    resp = response(point)
    coef = cl.spectral(resp,poly,point)
    
    model = cl.Expansion(coef,poly)
    modMc.append(model)
    
# %% Quadrature

order = 9
modQuad = []
quadList = []
nbrPts = int(2e4)
point = sampler(nbrPts)
ordList = [22,20,18,17,16,14,12,10]
polyTot = cl.gschmidt(ordList[0],point)
weight = 0

for i in range(len(ordList)):

    poly = copy.deepcopy(polyTot)
    poly.trunc(ordList[i])
    quadList.append(poly[:].shape[0])
    print(quadList[i])
    
    index,weight = cl.newquad(point,poly,weight)
    point = point[index]
    poly.trunc(order)
    
    resp = response(point)
    coef = cl.spectral(resp,poly,point,weight)
    
    model = cl.Expansion(coef,poly)
    modQuad.append(model)
    
# %% Error

nbrPts = int(1e5)
point = sampler(nbrPts)
resp = response(point)
errQuad = []
errMc = []

for i in range(len(ptList)):
    
    print('Error - ',i)
    respMc = modMc[i].eval(point)
    num = np.mean((resp-respMc)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    errMc.append(np.divide(num,den))
    
for i in range(len(quadList)):
    
    print('Error - ',i)
    respQuad = modQuad[i].eval(point)
    num = np.mean((resp-respQuad)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    errQuad.append(np.divide(num,den))

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.figure(1)
plt.loglog(ptList,errMc,'--C0',label='Quasi-MC')
plt.loglog(ptList,errMc,'.C0',markersize=size)
plt.semilogy(quadList,errQuad,'--C3',label='Quadrature')
plt.semilogy(quadList,errQuad,'.C3',markersize=size)
plt.ylabel('SRE [-]')
plt.xlabel('Points [-]')
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("ishigQuadMc.pdf",bbox_inches="tight",format="pdf",transparent=True)