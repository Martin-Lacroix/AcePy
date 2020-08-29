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

# quadList = [2300,1771,1330,1140,969,680,455,286]
# ptList = [251,398,1000,3162,10000,31622]
# errMc = [59.382579118262676,1.3101477888658908,0.01431766502735502,0.0003331323395077886,0.0001147689764197945,0.00010828667314923598]
# errQuad = [0.00010872653922746538,0.00010872918649873648,0.00011014953877830701,0.00011094149581626132,0.00022538520575371185,0.0030709843611228193,0.03594248747247988,0.1889033733260907]

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