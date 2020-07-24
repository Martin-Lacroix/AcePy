import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from pyro1d4 import response
import chaoslib as cl
import numpy as np

# %% Initialisation and PCE

order = 3
model = []
nbrPtsList = np.array([1e2,1e3,1e4,1e5,1e6]).astype(int)

for nbrPts in nbrPtsList:
    
    with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
    index = np.random.choice(point.shape[0],nbrPts,replace=0)
    point = point[index]
    resp = response(point)

    poly = cl.gschmidt(order,point)
    coef = cl.spectral(resp,poly,point)
    model.append(cl.Expansion(coef,poly))

# %% Error

nbrPts = int(1e5)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
meanMod = []
varMod = []
error = []

resp = response(point)
var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

for i in range(len(nbrPtsList)):

    print("Error",i)
    respMod = model[i].eval(point)
    meanMod.append(np.mean(respMod,axis=0))
    varMod.append(np.var(respMod,axis=0))

    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    error.append(np.divide(num,den))

# %% Figures

error = np.mean(error,axis=1)
stdMod = np.sqrt(varMod)
std = np.sqrt(var)

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

plt.figure(3,figsize=(8,4))
plt.loglog(nbrPtsList,error,'--C0',label="PCE")
plt.loglog(nbrPtsList,error,'.C0',markersize=size)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel("Mean SRE [-]")
plt.xlabel("Points [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro4ErrorMCpts.pdf",bbox_inches="tight",format="pdf",transparent=True)

# error = [33553.1,51.2747,0.22086,0.00400544,0.00228744]
# errorQuad = [3561.5608542126124,10.556600939535965,0.01047285897835352,0.003011628680087002,0.002242477142858169,0.0019541649624276996]
# nbrPtsListQuad = [35,210,715,1820,3876,7315]

# plt.figure(3)
# plt.loglog(nbrPtsList,error,'--C0',label="MC")
# plt.loglog(nbrPtsList,error,'.C0',markersize=size)
# plt.loglog(nbrPtsListQuad,errorQuad,'--C1',label="Fekete")
# plt.loglog(nbrPtsListQuad,errorQuad,'.C1',markersize=size)
# plt.legend()
# plt.ylabel("Mean SRE [-]")
# plt.xlabel("Points [-]")
# plt.grid(linewidth=width,color=grid)
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')
# plt.savefig("pyro4ErrorMCpts.pdf",bbox_inches="tight",format="pdf",transparent=True)