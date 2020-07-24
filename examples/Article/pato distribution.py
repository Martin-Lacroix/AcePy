import sys
sys.path.append('../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np

# %% Figures

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.axisbelow'] = True

size = 12
grid = [0.9,0.9,0.9]
width = 0.5
minct = 1

nbrPts = int(500)
with np.load("point.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

plt.figure(1,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.locator_params(axis="y",nbins=5)
plt.locator_params(axis="x",nbins=3)
plt.scatter(point[:,1],point[:,2],s=size)
plt.xlabel("$A_1$ [1/s]")
plt.ylabel("$E_1$ [J/mol]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoAE1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(2,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.locator_params(axis="y",nbins=5)
plt.scatter(point[:,5],point[:,6],s=size)
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoAE2.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(3,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.locator_params(axis="y",nbins=5)
plt.scatter(point[:,3],point[:,0],s=size)
plt.xlabel("$n_1$ [-]")
plt.ylabel("$F_1$ [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoNF1.pdf",bbox_inches="tight",format="pdf",transparent=True)

plt.figure(4,figsize=(3,3))
plt.ticklabel_format(axis='both',style='sci',scilimits=(0,0))
plt.locator_params(axis="y",nbins=5)
plt.scatter(point[:,7],point[:,4],s=size)
plt.xlabel("$n_2$ [-]")
plt.ylabel("$F_2$ [-]")
plt.grid(linewidth=width,color=grid)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("patoNF2.pdf",bbox_inches="tight",format="pdf",transparent=True)