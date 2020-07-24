import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Polynomial Chaos

nbrPts = int(2e4)
orders = [1,2,3,4,5]
trunc = [1,0.9,0.7,0.5]

with np.load("resp.npz",mmap_mode="r") as file: resp = file["resp"]
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]
resp = resp[index]

nbrTrunc = len(trunc)
nbrOrd = len(orders)
nbrPoly = np.zeros((nbrTrunc,nbrOrd))
model = []

for i in range(nbrTrunc):
    for j in range(nbrOrd):
    
        poly = cl.gschmidt(orders[j],point,trunc=trunc[i])
        nbrPoly[i,j] = poly[:].shape[0]

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

# %% Figures Coefficients

T = np.linspace(300,1400,101)
C = ["C0","C1","C2","C4"]
grid = [0.9,0.9,0.9]
width = 0.5
size = 8

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.figure(1,figsize=(8,4))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))

for i in range(nbrTrunc):

    C2 = '.'+C[i]
    C1 = '--'+C[i]
    plt.plot(orders,nbrPoly[i],C1,label="$q=$"+str(trunc[i]))
    plt.plot(orders,nbrPoly[i],C2,markersize=size)

plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0)
plt.ylabel("$n$ [-]")
plt.xlabel("Order [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8Hyper.pdf",bbox_inches="tight",format="pdf",transparent=True)