import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import numpy as np
import blue

# %% Parameters

nbrPts = int(1e4)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts)
point = point[index]

plt.rcParams["font.size"] = 16
plt.rcParams["legend.fontsize"] = 18
plt.rcParams["axes.unicode_minus"] = 0
plt.rcParams["font.family"] = "DejaVu Sans"

color = blue.colour()
minct = 1
grid = 60

# %% Distributions

plt.figure(1,figsize=(3,3))
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,0],point[:,1],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_1$ [1/s]")
plt.ylabel("$E_1$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8AE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2,figsize=(3,3))
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,2],point[:,3],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n_1$ [-]")
plt.ylabel("$F_1$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8NF1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,3],point[:,0],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$F_1$ [-]")
plt.ylabel("$A_1$ [1/s]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8FA1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,4],point[:,5],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8AE2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(5,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,6],point[:,7],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n_2$ [-]")
plt.ylabel("$F_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8NF2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(6,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,7],point[:,4],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$F_2$ [-]")
plt.ylabel("$A_2$ [1/s]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("pyro8FA2.pdf",bbox_inches="tight",format="pdf",transparent=True)