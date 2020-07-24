import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from pyro1d4 import response
from copy import deepcopy
import chaoslib as cl
import numpy as np

# %% Initialisation

order = 4
trunc = 1
model = []
iterations = [1,5,10,20,30,40,50,60,70]

nbrPts = int(2e5)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = response(point)

poly = cl.gschmidt(order,point,trunc=trunc)

# %% Coefficients

for it in iterations:

    poly2 = deepcopy(poly)
    coef,index = cl.lars(resp,poly2,point,it=it)
    coef = coef[index]
    poly2.clean(index)

    model.append(cl.Expansion(coef,poly2))

# %% Error

nbrPts = int(2e5)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = response(point)

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

# %% Figures

T = np.linspace(300,1400,101)
grid = [0.9,0.9,0.9]
width = 0.5
size = 8

error = np.mean(error,axis=1)
stdMod = np.sqrt(varMod)
std = np.sqrt(var)

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(3)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.semilogy(iterations,error,'--C0',label='LARS')
plt.semilogy(iterations,error,'.C0',markersize=size)
plt.ylabel("Mean SRE [-]")
plt.xlabel("Iterations [-]")
plt.grid(linewidth=width,color=grid)
plt.legend()
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyroErrorLars.pdf",bbox_inches="tight",format="pdf",transparent=True)