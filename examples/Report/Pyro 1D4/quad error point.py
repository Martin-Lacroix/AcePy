import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from pyro1d4 import response
import chaoslib as cl
import numpy as np

# %% Initialisation

nbrPts = int(1e5)
orders =  [3,6,9,12,15]

with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
nbrPtsList = []
model = []

for i in range(len(orders)):

    poly = cl.gschmidt(orders[i],point)
    nbrPtsList.append(poly[:].shape[0])
    print(poly[:].shape[0])

    index,weight = cl.fekquad(point,poly)
    
    poly.trunc(3)
    pointF = point[index]
    resp = response(pointF)

    coef = cl.spectral(resp,poly,pointF,weight)
    model.append(cl.Expansion(coef,poly))

# %% Error

nbrPts = int(1e5)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

error = []
varMod = []
meanMod = []

resp = response(point)
var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
std = np.sqrt(var)

for i in range(len(orders)):

    print("Error",i)

    respMod = model[i].eval(point)
    meanMod.append(np.mean(respMod,axis=0))
    varMod.append(np.var(respMod,axis=0))

    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    error.append(np.mean(np.divide(num,den)))
    
stdMod = np.sqrt(varMod)

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
plt.plot(T,mean,'C1',label="Ref")
plt.plot(T,meanMod[-1],'--C0',label="PCE")
plt.ylabel("Mean [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

plt.figure(2)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.plot(T,std,'C1',label="Ref")
plt.plot(T,stdMod[-1],'--C0',label="PCE")
plt.ylabel("SD [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# error = [196.63109753454023,3.0533579778794873,0.015916980391446862,0.0022613172114716133,0.001884232541460462]
# nbrPtsList = [35,210,715,1820,3876]

plt.figure(3,figsize=(8,4))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.semilogy(nbrPtsList,error,"C0--",label="PCE")
plt.semilogy(nbrPtsList,error,'.C0',markersize=size)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel("Mean SRE [-]")
plt.xlabel("Points [-]")
plt.ylim([1e-3,2e3])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro4ErrorQuad.pdf",bbox_inches="tight",format="pdf",transparent=True)

# plt.figure(3)
# plt.locator_params(axis='y',nbins=2)
# plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
# plt.semilogy(nbrPtsList,error,"C0--",label="Fekete points")
# plt.semilogy(nbrPtsList,error,'.C0',markersize=size)
# plt.legend()
# plt.ylabel("Mean SRE [-]")
# plt.xlabel("Points [-]")
# plt.ylim([1e-3,2e3])
# plt.grid(linewidth=width,color=grid)
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')
# plt.savefig("pyro4ErrorQuad.pdf",bbox_inches="tight",format="pdf",transparent=True)