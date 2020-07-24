import sys
sys.path.append('../../../')
from matplotlib import pyplot as plt
import chaoslib as cl
import numpy as np
import blue

# %% Function

def resc(point):
    
    T1 = 2000
    T2 = 1000
    
    A1 = point[:,1]
    E1 = point[:,2]
    A2 = point[:,5]
    E2 = point[:,6]
    A1 = np.log(A1)-E1/(8.314*T1)
    A2 = np.log(A2)-E2/(8.314*T2)
    E1 = E1/np.mean(E1)
    E2 = E2/np.mean(E2)
    
    ptRes = point.copy()
    ptRes[:,1] = A1
    ptRes[:,2] = E1
    ptRes[:,5] = A2
    ptRes[:,6] = E2
    
    return ptRes

# %% Probability Density

nbrPts = int(1e6)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)

color = blue.colour()
point = resc(point[index])
mapping = cl.Pca(point)
minct = 1
grid = 60

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.unicode_minus'] = 0
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'DejaVu Sans'

# %% Original Sample

nbrPts = int(1e4)
with np.load("pointALL.npz",mmap_mode="r") as file: point = file["pts"]
index = np.random.choice(point.shape[0],nbrPts,replace=0)
point = point[index]

plt.figure(1,figsize=(3,3))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.hexbin(point[:,5],point[:,6],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("$A_2$ [1/s]")
plt.ylabel("$E_2$ [J/mol]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("white1.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Param Sample

point = resc(point)

plt.figure(2,figsize=(3,3))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.hexbin(point[:,5],point[:,6],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("Param $A_2$ [-]")
plt.ylabel("Param $E_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("white2.pdf",bbox_inches="tight",format="pdf",transparent=True)

# %% Decorrelated Sample

point = mapping.white(point)
lim = [-2.5,2.5]

plt.figure(3,figsize=(3,3))
plt.ticklabel_format(axis='Y',style='sci',scilimits=(0,0))
plt.hexbin(point[:,5],point[:,6],gridsize=grid,cmap=color,mincnt=minct)
plt.xlabel("White $A_2$ [-]")
plt.ylabel("White $E_2$ [-]")
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.savefig("white3.pdf",bbox_inches="tight",format="pdf",transparent=True)