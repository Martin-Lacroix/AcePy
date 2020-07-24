import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
from decor1d4 import norm
import chaoslib as cl
import numpy as np
import blue

# %% Probability Density

dist = []
nbrPts = int(1e4)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
xPlot = np.linspace(-10,5,int(1e4))
point = norm(point[index])

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

color = blue.colour()
minct = 1
grid = 60

# Parameter A

mean = np.mean(point[:,0])
std = np.std(point[:,0])
dist.append(cl.Normal(mean,std))

plt.figure(1)
plt.plot(xPlot,dist[0].pdf(xPlot),"C0")
plt.hist(point[:,0],color="w",bins=40,ec="gray",density=True)
plt.ylabel("$f_A\,(A) [-]$")
plt.xlabel("$A$ [-]")
plt.xlim([mean-3.5*std,mean+3.5*std])

# Parameter E

mean = np.mean(point[:,1])
std = np.std(point[:,1])
dist.append(cl.Normal(mean,std))

plt.figure(2)
plt.plot(xPlot,dist[1].pdf(xPlot),"C0")
plt.hist(point[:,1],color="w",bins=40,ec="gray",density=True)
plt.ylabel("$f_E\,(E) [-]$")
plt.xlabel("$E$ [-]")
plt.xlim([mean-3.5*std,mean+3.5*std])

# Parameter n

mean = np.mean(point[:,2])
std = np.std(point[:,2])
dist.append(cl.Normal(mean,std))

plt.figure(3)
plt.plot(xPlot,dist[2].pdf(xPlot),"C0")
plt.hist(point[:,2],color="w",bins=40,ec="gray",density=True)
plt.ylabel("$f_n\,(n) [-]$")
plt.xlabel("$n$ [-]")
plt.xlim([mean-3.5*std,mean+3.5*std])

# Parameter F

mean = np.mean(point[:,3])
std = np.std(point[:,3])
dist.append(cl.Normal(mean,std))

plt.figure(4)
plt.plot(xPlot,dist[3].pdf(xPlot),"C0")
plt.hist(point[:,3],color="w",bins=40,ec="gray",density=True)
plt.axes().set_aspect('auto')
plt.ylabel("$f_F\,(F) [-]$")
plt.xlabel("$F$ [-]")
plt.xlim([mean-3.5*std,mean+3.5*std])

# %% Param Sample

plt.rcParams['font.size'] = 16

nbrPts = int(1e4)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = norm(point[index])
mapping = cl.Pca(point)

plt.figure(5,figsize=(3,3))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.hexbin(point[:,0],point[:,1],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("Param $A$ [-]")
plt.ylabel("Param $E$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("sampleNormAE.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Decorrelated Sample

point = mapping.white(point)
lim = [-2.5,2.5]

plt.figure(7,figsize=(3,3))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.hexbin(point[:,0],point[:,1],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("White $A$ [-]")
plt.ylabel("White $E$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("sampleDecAE.pdf",bbox_inches="tight",format="pdf",transparent=True)