import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from copy import deepcopy
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 4
trunc = 0.7
iterations = [5,20,40,60,80,100,120]
model = []

# %% Polynomial Computation

nbrPts = int(1e5)
with np.load("resp.npz",mmap_mode="r") as file: resp = file["resp"]
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

poly = cl.gschmidt(order,point,trunc=trunc)

# %% Coefficients

for it in iterations:

    poly2 = deepcopy(poly)
    coef,index = cl.lars(resp,poly2,point,it=it)

    coef = coef[index]
    poly2.clean(index)

    model.append(cl.Expansion(coef,poly2))

# %% Error

nbrPts = int(1e5)
with np.load("resp.npz",mmap_mode="r") as file: resp = file["resp"]
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

meanMod = []
varMod = []
error = []

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

for i in range(len(model)):

    print("Error",i)
    respMod = model[i].eval(point)
    meanMod.append(np.mean(respMod,axis=0))
    varMod.append(np.var(respMod,axis=0))

    num = np.mean((resp-respMod)**2,axis=0)
    den = np.mean(resp**2,axis=0)
    error.append(np.divide(num,den))

# %% Variation

modID = -1
nbrPts = int(1e6)

with np.load("resp.npz",mmap_mode="r") as file: resp = file["resp"]
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

resp = resp[index]
respMod = model[modID].eval(point)

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)

bound = np.sort(respMod,axis=0)
bound = bound[int(0.01*nbrPts):int(0.99*nbrPts)]
down = bound[0]
up = bound[-1]

# %% Figures

error = np.max(error,axis=1)
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
plt.fill_between(T,up,down,color='C0',alpha=0.2)
plt.plot(T,mean,'C1',label="Ref")
plt.plot(T,meanMod[modID],'--C0',label="PCE")
plt.legend(prop={'size':16})
plt.ylabel("g [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8Mean.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.plot(T,std,'C1',label="Ref")
plt.plot(T,stdMod[modID],'--C0',label="PCE")
plt.legend(prop={'size':16})
plt.ylabel("SD [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8Std.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(8,4))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.semilogy(iterations,error,'--C0',label='PCE')
plt.semilogy(iterations,error,'.C0',markersize=size)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel("Max SRE [-]")
plt.xlabel("Iterations [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8ErrorLars.pdf",bbox_inches="tight",format="pdf",transparent=True)