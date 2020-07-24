import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from pyro1d4 import response
import chaoslib as cl
import numpy as np

# %% Initialisation and PCE

order = 9
nbrPts = int(1e5)

with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

poly = cl.gschmidt(order,point)
index,weight = cl.fekquad(point,poly)

poly.trunc(3)
point = point[index]
resp = response(point)

coef = cl.spectral(resp,poly,point,weight)
model = cl.Expansion(coef,poly)

# %% Variation

nbrPts = int(1e6)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

respMod = model.eval(point)
resp = response(point)

var = np.var(resp,axis=0)
mean = np.mean(resp,axis=0)
meanMod = np.mean(respMod,axis=0)
varMod = np.var(respMod,axis=0)
stdMod = np.sqrt(varMod)
std = np.sqrt(var)

bound = np.sort(respMod,axis=0)
bound = bound[int(0.01*nbrPts):int(0.99*nbrPts)]
down = bound[0]
up = bound[-1]

# %% Figures

T = np.linspace(300,1400,101)
grid = [0.9,0.9,0.9]
width = 0.5

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.fill_between(T,up,down,color='C0',alpha=0.2)
plt.plot(T,mean,'C1',label="Ref")
plt.plot(T,meanMod,'--C0',label="PCE")
plt.ylabel("g [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro4MeanQuad.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2)
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.plot(T,std,'C1',label="Ref")
plt.plot(T,stdMod,'--C0',label="PCE")
plt.ylabel("SD [1/s]")
plt.xlabel("Temperature [K]")
plt.legend()
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro4StdQuad.pdf",bbox_inches="tight",format="pdf",transparent=True)