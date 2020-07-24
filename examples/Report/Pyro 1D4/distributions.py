import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import numpy as np
import blue

# %% Distributions

nbrPts = int(1e4)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts)
point = point[index]

plt.rcParams["font.size"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

color = blue.colour()
width = 0.5
minct = 1
grid = 60

plt.figure(1,figsize=(3,3))
plt.locator_params(axis='y', nbins=4)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,0],point[:,1],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A$ [1/s]")
plt.ylabel("$E$ [J/mol]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("sampleAE.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,2],point[:,3],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n$ [-]")
plt.ylabel("$F$ [-]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("sampleNF.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,3],point[:,0],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$F$ [-]")
plt.ylabel("$A$ [1/s]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("sampleFA.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Markov Chains

grid = [0.9,0.9,0.9]
color = "C0"
alpha = 0.5
width = 1
step = 80

nbrPts = int(5e4)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.arange(nbrPts)
index = index[0:-1:step]
point = point[index]

plt.figure(5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.plot(index,point[:,3],color=color,alpha=alpha,linewidth=width)
plt.plot(index,point[:,3],".",color=color)
plt.ylabel("F [-]")
plt.xlabel("Iteration [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("markovF.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(6)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.plot(index,point[:,1],color=color,alpha=alpha,linewidth=width)
plt.plot(index,point[:,1],".",color=color)
plt.ylabel("E [J/mol]")
plt.xlabel("Iteration [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("markovE.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(7)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.plot(index,point[:,0],color=color,alpha=alpha,linewidth=width)
plt.plot(index,point[:,0],".",color=color)
plt.ylabel("A [1/s]")
plt.xlabel("Iteration [-]")
plt.ylim([-0.05e6,0.8e6])
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("markovA.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(8)
plt.ticklabel_format(axis='X',style='sci',scilimits=(0,0))
plt.plot(index,point[:,2],color=color,alpha=alpha,linewidth=width)
plt.plot(index,point[:,2],".",color=color)
plt.ylabel("n [-]")
plt.xlabel("Iteration [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("markovN.pdf",bbox_inches="tight",format="pdf",transparent=True)