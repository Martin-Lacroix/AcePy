import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from decor1d4 import response,norm
import chaoslib as cl
import numpy as np

# %% Initialisation

orders = np.array([1,2,3,4,5])
dist = [cl.Normal(0,1) for i in range(4)]
ordQuad = 2*orders

nbrPts = int(1e6)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts)
point = norm(point[index])

mapping = cl.Pca(point)
model = []

# %% Polynomial Chaos

for i in range(len(orders)):

    point,weight = cl.tensquad(ordQuad[i],dist)
    poly = cl.polyrecur(orders[i],dist)
    resp = response(mapping.corr(point))
    coef = cl.spectral(resp,poly,point,weight)
    model.append(cl.Expansion(coef,poly))

# %% Monte Carlo and Error

nbrPts = int(1e5)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts)
point = norm(point[index])

meanMod = []
varMod = []
error = []

resp = response(point)
var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

for i in range(len(orders)):

    respMod = model[i].eval(mapping.white(point))
    meanMod.append(np.mean(respMod,axis=0))
    varMod.append(np.var(respMod,axis=0))
    
    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    error.append(np.divide(num,den))

error = np.array(error)
error = np.nanmean(error,axis=1)
stdMod = np.sqrt(varMod)
std = np.sqrt(var)
    
# %% Variation

nbrPts = int(1e6)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = norm(point[index])

respMod = model[3].eval(point)
resp = response(point)

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

bound = np.sort(resp,axis=0)
bound = bound[int(0.01*nbrPts):int(0.99*nbrPts)]
down = bound[0]
up = bound[-1]

# %% Figures

T = np.linspace(300,1400,101)
grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.fill_between(T,up,down,color='C0',alpha=0.2)
plt.plot(T,mean,'C1',label="Ref")
plt.plot(T,meanMod[3],'--C0',label="PCE")
plt.ylabel("g [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("resc4Mean.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.plot(T,std,'C1',label="Ref")
plt.plot(T,stdMod[3],'--C0',label="PCE")
plt.ylabel("SD [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("resc4Std.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(8,4))
plt.semilogy(orders,error,'--C0',label="PCE")
plt.semilogy(orders,error,'.C0',markersize=size)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel("Mean SRE [-]")
plt.xlabel("Order [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("resc4Error.pdf",bbox_inches="tight",format="pdf",transparent=True)