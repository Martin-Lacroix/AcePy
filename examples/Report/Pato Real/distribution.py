import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np
import blue

# %% Distribution Dec

dist = []
nbrPts = int(1e5)

dist.append(cl.Uniform(0.1,0.4))
dist.append(cl.Normal(1.2e4,1e3))
dist.append(cl.Normal(7.1e4,4e3))
dist.append(cl.Normal(3,0.2))
dist.append(cl.Uniform(0.1,0.3))
dist.append(cl.Normal(5e8,2e7))
dist.append(cl.Normal(1.7e5,6e3))
dist.append(cl.Normal(3,0.2))

dist = cl.Joint(dist)
point = dist.halton(nbrPts)

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.axisbelow'] = True

color = blue.colour()
gridColor = [0.9,0.9,0.9]
width = 0.5
minct = 1
grid = 60

plt.figure(1,figsize=(3,3))
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,1],point[:,2],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_1$ [1/s]")
plt.ylabel("$E_1$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoDecAE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2,figsize=(3,3))
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,5],point[:,6],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoDecAE2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,3],point[:,7],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n_1$ [-]")
plt.ylabel("$n_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoDecN12.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Distributions Real

nbrPts = int(1e3)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

plt.figure(4)
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,1],point[:,2],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_1$ [1/s]")
plt.ylabel("$E_1$ [J/mol]")

plt.grid(linewidth=width,color=gridColor)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("nonLinearAE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(5,figsize=(3,3))
plt.locator_params(axis="y",nbins=5)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,3],point[:,0],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n_1$ [-]")
plt.ylabel("$F_1$ [-]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCorNF1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(6,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,0],point[:,1],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$F_1$ [-]")
plt.ylabel("$A_1$ [1/s]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCorFA1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(7)
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,5],point[:,6],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")

plt.grid(linewidth=width,color=gridColor)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("nonLinearAE2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(8,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,7],point[:,4],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$n_2$ [-]")
plt.ylabel("$F_2$ [-]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCorNF2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(9,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.hexbin(point[:,4],point[:,5],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$F_2$ [-]")
plt.ylabel("$A_2$ [1/s]")

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoCorFA2.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Parameter

E1 = point[:,2]
E2 = point[:,6]
lnA1 = np.log(point[:,1])
lnA2 = np.log(point[:,5])

m1,p1 = np.polyfit(lnA1,E1,1)
m2,p2 = np.polyfit(lnA2,E2,1)

AE1 = E1/(m1*lnA1+p1)
AE2 = E2/(m2*lnA2+p2)

# plt.figure(10,figsize=(3,3))
# plt.locator_params(axis='both', nbins=4)
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.hexbin(lnA1,E1,gridsize=grid,cmap=color,mincnt=minct)
# plt.xlabel("ln$(A_1)$ [1/s]")
# plt.ylabel("$E_1$ [J/mol]")
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')
# plt.savefig("linearAE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(10)
plt.locator_params(axis='both', nbins=4)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.hexbin(lnA1,E1,gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("ln$(A_1)$ [1/s]")
plt.ylabel("$E_1$ [J/mol]")
plt.grid(linewidth=width,color=gridColor)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("linearAE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

# plt.figure(11,figsize=(3,3))
# plt.locator_params(axis='both', nbins=5)
# plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
# plt.hexbin(lnA2,E2,gridsize=grid,cmap=color,mincnt=minct)
# plt.xlabel("ln$(A_2)$ [1/s]")
# plt.ylabel("$E_2$ [J/mol]")
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')
# plt.savefig("linearAE2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(11)
plt.locator_params(axis='both',nbins=5)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.hexbin(lnA2,E2,gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("ln$(A_2)$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.grid(linewidth=width,color=gridColor)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("linearAE2.pdf",bbox_inches="tight",format="pdf",transparent=True)